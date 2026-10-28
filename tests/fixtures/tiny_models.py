"""Tiny, randomly initialised Hugging Face models saved to folders, so the inference worker can be
tested end to end without downloading anything. Their outputs are meaningless; their shapes,
files and code paths are those of the real models."""

from __future__ import annotations

from pathlib import Path

WORDS = ("the a cat dog sat on mat hello world speech image photo of red blue green car tree house person "
         "running jumping music digit zero one two three").split()


def tokenizer(folder: Path, *, bos: str = "<s>", eos: str = "</s>", chat: bool = False):
    from tokenizers import Tokenizer, models, pre_tokenizers, trainers
    from transformers import PreTrainedTokenizerFast

    tok = Tokenizer(models.BPE(unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    trainer = trainers.BpeTrainer(vocab_size=300, special_tokens=["<pad>", "<unk>", bos, eos],
                                  initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
    tok.train_from_iterator([" ".join(WORDS)] * 20, trainer)
    from tokenizers import decoders
    tok.decoder = decoders.ByteLevel()
    fast = PreTrainedTokenizerFast(tokenizer_object=tok, bos_token=bos, eos_token=eos, unk_token="<unk>",
                                   pad_token="<pad>")
    if chat:
        fast.chat_template = ("{% for m in messages %}<s>{{ m['role'] }}: {{ m['content'] }}</s>{% endfor "
                              "%}{% if add_generation_prompt %}<s>assistant:{% endif %}")
    fast.save_pretrained(folder)
    return fast


def causal_lm(folder: Path) -> Path:
    from transformers import LlamaConfig, LlamaForCausalLM

    tok = tokenizer(folder, chat=True)
    config = LlamaConfig(vocab_size=len(tok), hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                         num_attention_heads=2, num_key_value_heads=2, max_position_embeddings=128,
                         bos_token_id=tok.bos_token_id, eos_token_id=tok.eos_token_id,
                         pad_token_id=tok.pad_token_id)
    LlamaForCausalLM(config).save_pretrained(folder)
    return folder


def encoder(folder: Path) -> Path:
    from transformers import BertConfig, BertModel

    tok = tokenizer(folder)
    BertModel(BertConfig(vocab_size=len(tok), hidden_size=32, num_hidden_layers=2, num_attention_heads=2,
                         intermediate_size=64, max_position_embeddings=128)).save_pretrained(folder)
    return folder


def clip(folder: Path) -> Path:
    from transformers import CLIPConfig, CLIPImageProcessor, CLIPModel

    tok = tokenizer(folder)
    config = CLIPConfig(text_config={"vocab_size": len(tok), "hidden_size": 32, "intermediate_size": 64,
                                     "num_hidden_layers": 2, "num_attention_heads": 2,
                                     "max_position_embeddings": 64,
                                     "projection_dim": 16, "eos_token_id": tok.eos_token_id,
                                     "pad_token_id": tok.pad_token_id, "bos_token_id": tok.bos_token_id},
                        vision_config={"hidden_size": 32, "intermediate_size": 64, "num_hidden_layers": 2,
                                       "num_attention_heads": 2, "image_size": 32, "patch_size": 8,
                                       "projection_dim": 16},
                        projection_dim=16)
    CLIPModel(config).save_pretrained(folder)
    CLIPImageProcessor(size={"shortest_edge": 32}, crop_size={"height": 32,
                                                              "width": 32}).save_pretrained(folder)
    return folder


def whisper(folder: Path) -> Path:
    from transformers import WhisperConfig, WhisperFeatureExtractor, WhisperForConditionalGeneration

    tok = tokenizer(folder, bos="<|startoftranscript|>", eos="<|endoftext|>")
    config = WhisperConfig(vocab_size=len(tok), d_model=32, encoder_layers=1, decoder_layers=1,
                           encoder_attention_heads=2, decoder_attention_heads=2, encoder_ffn_dim=64,
                           decoder_ffn_dim=64, num_mel_bins=80, max_source_positions=1500,
                           max_target_positions=64,
                           decoder_start_token_id=tok.bos_token_id, eos_token_id=tok.eos_token_id,
                           pad_token_id=tok.pad_token_id, bos_token_id=tok.bos_token_id,
                           begin_suppress_tokens=None, suppress_tokens=None)
    model = WhisperForConditionalGeneration(config)
    model.generation_config.decoder_start_token_id = tok.bos_token_id
    model.generation_config.forced_decoder_ids = None
    model.save_pretrained(folder)
    WhisperFeatureExtractor(feature_size=80).save_pretrained(folder)
    return folder


def detector(folder: Path) -> Path:
    from transformers import YolosConfig, YolosForObjectDetection, YolosImageProcessor

    config = YolosConfig(hidden_size=32, num_hidden_layers=2, num_attention_heads=2, intermediate_size=64,
                         image_size=[64, 64], patch_size=16, num_detection_tokens=5, num_labels=3,
                         id2label={0: "cat", 1: "dog", 2: "car"}, label2id={"cat": 0, "dog": 1, "car": 2})
    YolosForObjectDetection(config).save_pretrained(folder)
    YolosImageProcessor(size={"shortest_edge": 64, "longest_edge": 64}).save_pretrained(folder)
    return folder


def blip(folder: Path) -> Path:
    from transformers import BlipConfig, BlipForConditionalGeneration, BlipImageProcessor

    tok = tokenizer(folder, bos="[DEC]", eos="[SEP]")
    config = BlipConfig(text_config={"vocab_size": len(tok), "hidden_size": 32, "intermediate_size": 64,
                                     "num_hidden_layers": 2, "num_attention_heads": 2,
                                     "max_position_embeddings": 64,
                                     "encoder_hidden_size": 32, "bos_token_id": tok.bos_token_id,
                                     "eos_token_id": tok.eos_token_id, "pad_token_id": tok.pad_token_id,
                                     "sep_token_id": tok.eos_token_id},
                        vision_config={"hidden_size": 32, "intermediate_size": 64, "num_hidden_layers": 2,
                                       "num_attention_heads": 2, "image_size": 32, "patch_size": 8})
    model = BlipForConditionalGeneration(config)
    model.config.text_config.bos_token_id = tok.bos_token_id
    model.save_pretrained(folder)
    BlipImageProcessor(size={"height": 32, "width": 32}).save_pretrained(folder)
    return folder
