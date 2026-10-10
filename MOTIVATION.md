# Why Text-as-Images Changes Everything

**What if the root cause of most LLM failures is how we tokenize text?**

Visually encoding text mimics how humans read, and creates equivalence
between what the human sees and how the computer processes the text.

Pre-tokenization into meaningful units (e.g. words) allows the model to encode information across languages
more equally, and reduces the impact of tokenization artifacts.

## Tokenization

In his [lecture](https://www.youtube.com/watch?v=zduSFxRajkE), Andrej Karpathy discusses weird behaviors in models
that trace back to tokenization.

- Why can't LLM spell words? **Tokenization**.
- Why can't LLM do super simple string processing tasks like reversing a string? **Tokenization**.
- Why is LLM worse at non-English languages (e.g. Japanese)? **Tokenization**.
- Why is LLM bad at simple arithmetic? **Tokenization**.
- Why did GPT-2 have more than necessary trouble coding in Python? **Tokenization**.
- Why did my LLM abruptly halt when it sees the string "<endoftext>"? **Tokenization**.
- What is this weird warning I get about a "trailing whitespace"? **Tokenization**.
- Why the LLM break if I ask it about "SolidGoldMagikarp"? **Tokenization**.
- Why should I prefer to use YAML over JSON with LLMs? **Tokenization**.
- Why is LLM not actually end-to-end language modeling? **Tokenization**.
- What is the real root of suffering? **Tokenization**.

What if we encoded text as images of pre-tokenized words (alongside bytes)?
Each word would be embedded on its own, from what it looks like and how it is spelled, with no subword tokenizer
or vocabulary.

- ✅ LLMs should be able to spell words, they see the characters.
- ✅ LLMs should be able to do string processing tasks, they see the characters.
- ✅ LLMs should be equally good at all languages, with equitable pre-tokenization.
- ❌ Unclear if LLMs will be better at arithmetic, but they should be able to see the numbers.
- ❌ Unclear if LLMs will be better at coding.
- ✅ LLMs would not abruptly halt on special tokens, as the special tokens are different images.
- ✅ No more warnings about trailing whitespace, as whitespace is part of the token.
- ✅ LLMs should not break on weird words, as the tokenizer is not trained separately.
- ✅ JSON and YAML should be equally easy, as the quote-marks are part of the token.
- ☑️ LLMs should be more end-to-end language modeling *except for pre-tokenization*.
- ❓ The real root of suffering is still unclear.

## Robust Open Vocabulary Translation

[Salesky et al. (2021)](https://arxiv.org/pdf/2104.08211) claim that:

> Machine translation models have discrete vocabularies and commonly use subword segmentation techniques
> to achieve an ‘open vocabulary.’ This approach relies on consistent and correct underlying unicode sequences,
> and makes models susceptible to degradation from common types of noise and variation.

<img alt="Examples of common behavior which cause divergent representations for subword models" src="./assets/phenomena.png" width="500px">

### Diacritics

For latin scripts, such as German, we may use diacritics such as Umlauts (e.g. `ä`, `ö`, `ü`).
We can write them down either as a single character (e.g. Unicode Normalization Form C `ü` = [2448]),
or as a combination of two characters (e.g. Unicode Normalization Form D `u` + `¨` = [84, 136, 230]).

The paper gives the example of Arabic `كتاب` (with 3 tokens in GPT-4)  which fully vowelized is `كِتَابٌ` (7 tokens).
Another would be Hebrew `ספר` (with 5 tokens in GPT-4) which diacritized is `סֵפֶר` (9 tokens).

### Misspelling

The paper gives an example of the words `language` and `langauge` which are tokenized as 1 and 2 tokens
respectively in GPT-4, giving a very different representation to what is likely intended as the same meaning.
The problems may only increase in non-latin scripts.

### Visually Similar / Identical Characters

People often obfuscate text using visually similar or identical characters,
with homograph attacks --- using characters that look the same from different scripts,
to LeetSpeak --- using characters that look similar.

For example, the Latin character `a` (U+0061) looks very similar to the Cyrillic character `а` (U+0430).
Given the word `man` (1 token in GPT-4), if we replace the `a` with the Cyrillic `а`, we get `mаn` (3 tokens).

The paper gives the LeetSpeak example for `really` vs `rea11y` (1 token vs 3 tokens in GPT-4).
