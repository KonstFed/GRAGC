from typing import Literal, Any, Dict

from pydantic import Field
import torch
from tqdm import trange
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from ragc.graphs.common import Node
from ragc.llm.embedding import BaseEmbedder, BaseEmbederConfig
from ragc.llm.generator import BaseGenerator, BaseGeneratorConfig, AugmentedGenerator, AugmentedGeneratorConfig


class HuggingFaceEmbedder(BaseEmbedder):
    def __init__(self, model_name: str, max_batch_size: int, max_length: int | None, store_in_gpu: bool = False):
        self.model_name = model_name
        self.max_batch_size = max_batch_size
        self.store_in_gpu = store_in_gpu

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

        if max_length is None:
            self.max_length = self.tokenizer.model_max_length
        else:
            self.max_length = max_length

        if store_in_gpu:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model.to(device)

    def embed(self, inputs: list[str] | str) -> torch.Tensor:
        if isinstance(inputs, str):
            inputs = [inputs]

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)

        batch_size = min(self.max_batch_size, len(inputs))
        embeddings = []

        for i in trange(0, len(inputs), batch_size):
            batch = inputs[i : min(i + batch_size, len(inputs))]
            tokenized_inputs = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=self.max_length,
            ).to(device)
            with torch.no_grad():
                outputs = self.model(**tokenized_inputs)

            embeddings.append(outputs.last_hidden_state.mean(dim=1))

        embeddings = torch.cat(embeddings)

        if not self.store_in_gpu:
            embeddings = embeddings.cpu()
            self.model.to(torch.device("cpu"))

        return embeddings


class HuggingFaceEmbedderConfig(BaseEmbederConfig):
    type: Literal["hugging_face"] = "hugging_face"

    model_name: str
    max_batch_size: int
    max_length: int | None = None
    store_in_gpu: bool = True

    def create(self) -> HuggingFaceEmbedder:
        return HuggingFaceEmbedder(
            model_name=self.model_name,
            max_batch_size=self.max_batch_size,
            max_length=self.max_length,
            store_in_gpu=self.store_in_gpu,
        )


class HuggingFaceGenerator(BaseGenerator):
    def __init__(self, model: str, tokenizer_kwargs: dict, generation_kwargs: dict):
        self.model_name = model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = AutoModelForCausalLM.from_pretrained(model, device_map="auto")
        self.model.to(self.device)

        self.tokenizer = AutoTokenizer.from_pretrained(model, padding_side="left")

        self._tokenizer_kwargs = tokenizer_kwargs
        self._generation_kwargs = generation_kwargs

    def generate(self, prompt: str) -> str:
        model_inputs = self.tokenizer([prompt], return_tensors="pt", **self._tokenizer_kwargs).to(self.device)
        input_length = model_inputs.input_ids.shape[1]

        generated_ids = self.model.generate(**model_inputs, **self._generation_kwargs)
        generated_ids = generated_ids[0, input_length:]
        return self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]


class HuggingFaceGeneratorConfig(BaseGeneratorConfig):
    type: Literal["hugging_face"] = "hugging_face"
    model: str

    tokenizer_kwargs: dict[str, Any] = Field(default_factory=dict)
    generation_kwargs: dict[str, Any] = Field(default_factory=dict)

    def create(self) -> HuggingFaceGenerator:
        return HuggingFaceGenerator(
            model=self.model,
            tokenizer_kwargs=self.tokenizer_kwargs,
            generation_kwargs=self.generation_kwargs,
        )


class Singleton(type):
    _instances = {}
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class CompletionGenerator(AugmentedGenerator, metaclass=Singleton):
    def __init__(
        self,
        model_path: str,
        max_input: int,
        max_gen: int,
        local_context_lines: int,
        generation_args: Dict[str, Any],
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            truncation_side='left'
        )
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            # torch_dtype=torch.bfloat16,
            quantization_config=quantization_config,
            device_map=self.device,
        )

        # truncation settings
        self.max_input = max_input
        self.max_gen = max_gen
        self.local_context_lines = local_context_lines

        # generation arguments
        self.generation_args = generation_args

        # debug variable
        self.trunc_inputs = 0
        self.trunc_gens = 0

    def __align(self, generation: str, completion_path: str) -> str:
        # --- find start_ix ---
        # try to find required namespace completion
        ix = generation.rfind(completion_path)
        if ix != -1:
            generation = generation[ix:]
        else:
            print('WARNING: Alignment failed - completion not found!')
        
        # remove signature
        lines = generation.split('\n')
        start_ix = 0
        while start_ix < len(lines) - 1 and not lines[start_ix].startswith('    '):
            start_ix += 1
    
        # --- find end_ix by searching zero indent ---
        end_ix = start_ix + 1
        while end_ix < len(lines) and (lines[end_ix] == '' or lines[end_ix].startswith(' ')):
            end_ix += 1
    
        # strip end tokens
        completion = '\n'.join(lines[start_ix:end_ix]).strip('\n')
        
        return completion

    def __prompt(self, prompt: str, relevant_nodes: list[Node], local_context: str, completion_path: str) -> str:
        # sort by length
        relevant_nodes = sorted(relevant_nodes, key=lambda x: -len(x.code))

        # cross context
        docs = []
        for node in relevant_nodes:
            docs.append(f'#{node.file_path}\n{node.code}')

        # local context
        if self.local_context_lines > 0 and local_context:
            local_lines = local_context.split('\n')
            local_context = '\n'.join(local_lines[max(0, len(local_lines) - self.local_context_lines):])
            local_context_prompt = f'#{completion_path}\n{local_context}'
        else:
            local_context_prompt = ""

        return '\n\n'.join(docs + [local_context_prompt] + [prompt])

    def generate(self, query: str, relevant_nodes: list[Node]) -> str:
        try:
            # unpack query
            prompt = query['prompt']
            local_context = query['local_context']
            completion_path = query['completion_path']
            
            # encode prompt
            full_prompt = self.__prompt(prompt, relevant_nodes, local_context, completion_path)
            inputs = self.tokenizer(
                full_prompt,
                truncation=True,
                max_length=self.max_input,
                return_tensors="pt"
            ).to(self.device)

            # debug info
            input_len = inputs['input_ids'].shape[1]
            if input_len >= self.max_input:
                print('WARNING: truncating input!')
                self.trunc_inputs += 1

            # generation
            outputs = self.model.generate(
                **inputs,
                tokenizer=self.tokenizer,
                max_new_tokens=self.max_gen,
                stop_strings=[f'\n{chr(i)}' for i in range(ord('!'), ord('~') + 1)],
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
                **self.generation_args,
            )

            # debug info
            output_len = outputs.shape[1]
            if (output_len - input_len) >= self.max_gen:
                print('WARNING: truncating generation!')
                self.trunc_gens += 1

            # decoding and alignment
            generation = self.tokenizer.decode(outputs[0])
            completion = self.__align(generation, completion_path=query['completion_path'])
            print('full_prompt', full_prompt)
            print('generation', generation)
            print('completion', completion)

            return completion
        except torch.OutOfMemoryError:
            msg = f"OutOfMemoryError with {len(inputs['input_ids'][0])} input tokens"
            print(f"WARNING: {msg}!")
            return f"raise NotImplementedError('{msg}')"
        finally:
            print(f'trunc_inputs:', self.trunc_inputs)
            print(f'trunc_gens  :', self.trunc_gens)


class CompletionGeneratorConfig(AugmentedGeneratorConfig):
    type: Literal["completion"] = "completion"
    model_path: str
    max_input: int = 16_354
    max_gen: int = 4096
    local_context_lines: int = 0
    generation_args: Dict[str, Any] = {}

    def create(self) -> CompletionGenerator:
        return CompletionGenerator(
            model_path=self.model_path,
            max_input=self.max_input,
            max_gen=self.max_gen,
            local_context_lines=self.local_context_lines,
            generation_args=self.generation_args,
        )
