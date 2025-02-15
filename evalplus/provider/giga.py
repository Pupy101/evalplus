from multiprocessing.pool import ThreadPool
from typing import List

from evalplus.gen.util import giga_request
from evalplus.provider.base import DecoderBase


class GigaDecoder(DecoderBase):
    def __init__(self, name: str, **kwargs) -> None:
        super().__init__(name, **kwargs)
        self.client = giga_request.GigaChat()

    def codegen(
        self, prompt: str, do_sample: bool = True, num_samples: int = 200
    ) -> List[str]:
        if do_sample:
            assert self.temperature > 0, "Temperature must be positive for sampling"

        batch_size = min(self.batch_size, num_samples)
        if not do_sample:
            assert batch_size == 1, "Sampling only supports batch size of 1"

        outputs = []

        def _request(_) -> str:
            message = giga_request.make_auto_request(
                client=self.client,
                model=self.name,
                messages=[
                    {
                        "role": "user",
                        "content": self.instruction_prefix
                        + f"\n```python\n{prompt.strip()}\n```\n",
                    }
                ],
                max_tokens=self.max_new_tokens,
                temperature=self.temperature,
            )
            return message["choices"][0]["message"]["content"]

        if batch_size <= 1:
            return [_request(None)]

        with ThreadPool(processes=batch_size) as pool:
            outputs = list(pool.map(_request, range(batch_size)))

        return outputs

    def is_direct_completion(self) -> bool:
        return False
