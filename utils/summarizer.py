from functools import wraps
from typing import List, Tuple
import torch
import transformers


class Summmarizer:
    _defaults = {"model_type": "bart", "model_name": "facebook/bart-large-cnn"}

    def __init__(self, model_type=None, model_name=None, device=None):
        self._model = None
        self._tokenizer = None
        self._max_tokens = None
        self._device = None

        self._tot_input_tokens = 0
        self._tot_output_tokens = 0

        if device:
            self._device = device
        else:
            self._device = "cuda:0" if torch.cuda.is_available() else "cpu"

        if not model_type:
            model_type = self._defaults["model_type"]
        if not model_name:
            model_name = self._defaults["model_name"]

        (
            self._model,
            self._tokenizer,
            self._max_tokens,
        ) = self.__class__.init_model_tokenizer(
            model_type, model_name, self._device
        )

    @staticmethod
    def init_model_tokenizer(
        model_type: str, model_name: str, device: str
    ) -> Tuple[
        transformers.PreTrainedModel, transformers.PreTrainedTokenizer, int
    ]:
        """
        Initilize model and tokenizer for summarization

        :param model_type: str for type of model ("bart" or "t5)
        :param model: name of model to use, must correspond with model type (eg. 'facebook/bart-large-cnn')
        :return: tuple containing the model, tokenizer, and max length
        """

        if model_type == "bart":
            # create BART
            from transformers import BartTokenizer, BartForConditionalGeneration

            model = BartForConditionalGeneration.from_pretrained(model_name).to(
                device
            )
            tokenizer = BartTokenizer.from_pretrained(model_name)
            max_tokens = 1024
        elif model_type == "t5":
            # create T5
            from transformers import T5ForConditionalGeneration, T5Tokenizer

            model = T5ForConditionalGeneration.from_pretrained(model_name).to(
                device
            )
            tokenizer = T5Tokenizer.from_pretrained(model_name)
            max_tokens = 512
        else:
            raise ValueError(f"Invalid model type: '{model_type}'")

        return model, tokenizer, max_tokens

    def summarize(
        self,
        text: str,
        low: float = 0.1,
        high: float = 0.5,
    ) -> str:
        """
        Summarize the input string. To help contain context, it is useful to split
        content into paragraphs (newline seperated)

        :param text: string to summarize
        :param low: minimum length multiplier, defaults to 0.1 which signifies 10% of original len
        :param high: maximum length multiplier, defaults to 0.5 which signifies 50% of original len
        :return: summary of text
        """
        inputs = self.batch_tokenize(text)
        if inputs:
            output = self.gen_batch(inputs, low, high)
            return "\n".join(output)
        else:
            return ""

    @wraps(summarize)
    def __call__(self, *args, **kwargs):
        return self.summarize(*args, **kwargs)

    def batch_tokenize(self, text: str) -> List[torch.tensor]:
        """
        Tokenize string and create batches. Batches try to keep context as complete
        as possible by avoiding splitting paragraphs whenever possible.

        :param text: text to tokenize
        :return: list of batched token tensors
        """
        # split into paragraphs to help ensure related content is summarized together
        pars = text.split("\n")

        # tokenize paragraphs
        # disable truncation and allow pars longer than max for model
        inputs = [
            self._tokenizer.encode(
                par, return_tensors="pt", max_length=None, truncation=False
            ).to(self._device)
            for par in pars
            if (not par.isspace()) and (par)
        ]

        if not inputs:
            return inputs

        # precompute encoded paragraph tensor sizes
        sizes = [in_tensor.size()[1] for in_tensor in inputs]

        new_inputs = []
        skip_ind = set()
        # iterate through tokenized paragraphs
        for in_par_ind in range(len(sizes)):
            if in_par_ind in skip_ind:
                continue

            # get encoded paragraph tensor
            in_par = inputs[in_par_ind]
            # get encoded paragraph tensor len
            in_par_size = sizes[in_par_ind]

            # check if we should split or combine
            if in_par_size > self._max_tokens:
                # split paragraph
                in_par_list = [
                    res.to(self._device)
                    for res in torch.split(in_par, self._max_tokens - 1, dim=1)
                ]

                new_inputs.extend(in_par_list)
            else:
                # combine paragraphs

                # combine with next until full or reach end
                for in_par_next_ind in range(in_par_ind + 1, len(inputs)):

                    # get next encoded paragraph tensor
                    in_par_next = inputs[in_par_next_ind]
                    # get next encoded paragraph tensor len
                    in_par_next_size = sizes[in_par_next_ind]

                    # check if we can combine
                    if in_par_size + in_par_next_size <= self._max_tokens - 1:
                        # concat curr and next encoded paragraph tensors
                        in_par = torch.cat((in_par, in_par_next), 1)
                        in_par_size = in_par.size()[1]

                        # mark to skip next par in outer loop
                        skip_ind.add(in_par_next_ind)
                    else:
                        # break so we don't combine out of order paragraphs
                        break

                new_inputs.append(in_par.to(self._device))

        return new_inputs

    def gen_batch(
        self,
        batch: List[torch.tensor],
        low: float = 0.1,
        high: float = 0.5,
    ) -> List[str]:
        """generate summaries from batch of encoded strings

        :param batch: list of tokenized string tensors
        :param low: minimum length multiplier, defaults to 0.1 which signifies 10% of original len
        :param high: minimum length multiplier, defaults to 0.5 which signifies 50% of original len
        :return: decoded text of batches
        """

        # generate tokenized output tensors from input tensors
        outputs = [
            self._model.generate(
                input_tensor,
                max_length=round((input_tensor.size()[1] - 1) * high),
                min_length=round((input_tensor.size()[1] - 1) * low),
                length_penalty=1.0,
                do_sample=True,
                num_beams=4,
            )
            for input_tensor in batch
        ]

        # decode and combine output into string
        pars_text_out = [
            " ".join(
                [
                    self._tokenizer.decode(out_id, skip_special_tokens=True)
                    for out_id in output
                ]
            )
            for output in outputs
        ]

        # calculate stats
        tot_in = sum([input_tensor.size()[1] for input_tensor in batch])
        tot_out = sum([out.size()[1] for out in outputs])

        # find agg % reduced
        self._tot_input_tokens += tot_in
        self._tot_output_tokens += tot_out

        return pars_text_out

    def print_stats(self):
        """
        Print overall statistics for all text summarized
        """

        perc_red = round(
            (self._tot_input_tokens - self._tot_output_tokens)
            * 100
            / self._tot_input_tokens
        )
        print(
            f"{self._tot_input_tokens} -> {self._tot_output_tokens} ({perc_red}% reduced)"
        )


if __name__ == "__main__":
    text = (
        "A Georgia judge issued a tongue-in-cheek order banishing the Elf "
        "on the Shelf -- a recent Christmas tradition -- from his county. Cobb"
        " County Superior Court Judge Robert Leonard tweeted the text of an "
        "order banishing the Elf on a Shelf, a small toy elf that reports the "
        "actions of children to Santa Claus in the runup to Christmas, due to "
        "it posing \"a risk to the emotional health and well being of Cobb's "
        'young children." "Inexplicably, Elves sometimes move and don\'t move '
        "overnight. When these Elves do not move, it leaves our children of "
        'tender years in states of extreme emotional distress," Leonard wrote.'
        " The Elf on the Shelf sprang out of a 2005 children's book of the "
        "same name by Carol Aebersold and Chanda Bell. Leonard tweeted that "
        'his order was a "gift to tired parents." He explained that families '
        "who love their elves can feel free to keep them."
    )
    summarizer = Summmarizer()
    summary = summarizer(text)
    print(summary)
    summarizer.print_stats()
