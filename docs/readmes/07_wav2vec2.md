# Wav2Vec2

This module fine-tunes Facebook's `Wav2Vec2` model for automatic speech recognition (ASR) using movie subtitles. The model learns to map raw audio waveforms to transcribed text using self-supervised representations and Connectionist Temporal Classification (CTC) loss.

## Motivation

Wav2Vec2 provides strong performance in low-resource ASR tasks by leveraging unsupervised pretraining on raw audio. Our motivation is to adapt it to real-world, noisy movie dialogue, which includes diverse acoustic conditions such as music, overlapping speech, and varying audio quality.

## Architecture

![Wav2Vec2 Framework](../assets/wav2vec2_framework.png)
> llustration of Wav2Vec2 framework which jointly learns contextualized speech representations
and an inventory of discretized speech units.

### Feature Encoder

The feature encoder consists of a series of temporal convolutional layers, each followed by:

- **Layer Normalization**
- **GELU Activation**

### Contextualized representations with Transformers

The encoder outputs are passed through a **contextual network** built on the Transformer architecture. Instead of fixed positional encodings, Wav2Vec2 uses a convolutional layer for **relative positional information**, followed by:

- **GELU Activation**
- **Layer Normalization**

### Quantization Module

For self-supervised pretraining, the model discretizes encoder outputs `z` into a finite vocabulary of speech units using **product quantization**. These are compared with predictions from the Transformer to compute contrastive loss.

## Preprocessing

We used the built-in Wav2Vec2 processor from Hugging Face for audio and text preprocessing.

### Audio Preprocessing Steps

The following were done before feeding into Wav2Vec2 processor.

- **Downmixing**: Stereo audio is converted to mono by averaging channels.
- **Normalization**: Waveforms are scaled to the range `[-1, 1]` to stabilize training.
- **Resampling**: All audio is resampled to **16 kHz**, per the [Nyquist-Shannon Theorem](https://en.wikipedia.org/wiki/Nyquist%E2%80%93Shannon_sampling_theorem), to preserve speech information while reducing data size.

### Subtitle Preprocessing

- Spaces `" "` are replaced with `"|"` to mark word boundaries.
- Newlines (`"\n"`) are removed.
- All text is converted to **uppercase**.

## Training Setup

Training was conducted using Hugging Face's `Trainer` API.

### Model: 

```python
model = Wav2Vec2ForCTC.from_pretrained(
    "facebook/wav2vec2-base-960h",
    vocab_size=len(processor.tokenizer), # 32 in this case
    ctc_loss_reduction="mean",
    pad_token_id=processor.tokenizer.pad_token_id,
    ctc_zero_infinity=True, # zeroes infinite losses
)
```

### Data Collator

We implemented a custom collator to:

- Pad audio inputs
- Pad text token IDs and replace padding tokens with `-100` (to ignore in loss)


### Training Args

```python
output_dir="./wav2vec2-finetuned-moviesubs",
group_by_length=True,
dataloader_num_workers=4,
per_device_train_batch_size=2,
eval_strategy="epoch",
save_strategy="epoch",
load_best_model_at_end=True,
num_train_epochs=80,
fp16=False,
learning_rate=1e-7,
logging_strategy="steps",
logging_dir="./logs",
logging_steps=10,
report_to="none",
save_total_limit=1,
remove_unused_columns=False,
max_grad_norm=0.05,
```

- **Batch size = 2** due to GPU memory constraints.
- **Learning rate = 1e-7** to mitigate instability during fine-tuning.
- **Gradient clipping** (max_grad_norm = 0.05) to prevent exploding gradients.
- **fp16 disabled** due to numerical instability.
- **Early stopping** was included (but not used in the 80-epoch run).

### Trainer

```python
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    tokenizer=processor,
    data_collator=data_collator,
    compute_metrics=wrapped_compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=10)],
)
```

The model improved from:

- Validation Loss: 1.9520 → 1.7371
- WER (Word Error Rate): 0.6464 → 0.6132

## Loss Curves

![Training vs Validation Loss](../../plots/wav2vec2_train_vs_val_loss_dark.png)

Both training and validation loss decrease together, indicating successful learning. Training loss stabilizes around **2.5**, and validation around **1.7**.

## Predictions

### Test Data

The test set consists of **five hand-picked audio segments** from *Once Upon a Time in Hollywood (2019)*, directed by Quentin Tarantino. These samples were selected to represent a variety of real-world challenges for speech recognition systems, including:

- **Multiple speakers** with distinct voices and accents
- **Background noise** such as music, ambient sounds, and crowd noise
- **Variability in audio quality**, volume, and clarity

This diverse selection provides a meaningful benchmark for evaluating the model’s robustness under noisy, multi-speaker, and acoustically complex conditions.

📁 **Access the test data here:** [**test**](../../data/test/)

### segment_000200.wav

Context: Screaming woman in background, two speakers.

**Ground Truth**:

```
how'd your little talk with george go are we kidnapping him not the word i'd use now you've talked to him you believe everything's all right not exactly this was a mistake you should leave way ahead of you george isn't blind you're the blind one girls yelling indistinctly
```

**Prediction**:

```
hav your little talk with george to go ar we kid nappingham not the word i'd use well now tht you've talkd tohome do you believe everything's all right not exactly this was a mistake you should leave wher  george and blind ger blind blind
```

#### segment_000184.wav

Context: Clean audio, good volume.

**Ground Truth**:

```
you the mama bear can i help you i hope so i'm an old friend of george's thought i'd stop and say hello that's very nice of you but you picked the wrong time george is taking a nap right now oh that is unfortunate yes it is what's your name cliff booth how do you know george i used to shoot westerns here at the ranch when was the last time you saw george
```

**Prediction**:

```
mama bear can i helpe you i hope so i'm an old friend of georges thought i stop say hallo it's very nice of you but unfortunately picked o the wrong time george's taking a man byto oh that is unfortuyes itis what's your name cliff bot had he know george i used to shot westrn sheere at the randage when i was last time you saw a george
```

#### segment_000047.wav

Context: Minimal background noise, low volume.

**Ground Truth**:

```
growls peggy on tv i waited at the bar till closing time but he never came back man on tv okay peggy what happened peggy i don't know everything was fine we had dinner at my house and afterwards you know while i was doing the dishes tsk tsk whines he and tobey played and then at the club gabe was doing great then wham a sudden change you know how musicians are they're temperamental cats who knows what got into him yeah
```

**Prediction**:

```
h  wis  oto losing time hou never came back your kit jagi wa i don't know everything was fine  o and after it's dot young while i doing the dishesye   dhe was n ra and we sudden change you know how musicians either temperamental cats who knows what got into o
```

#### segment_000096.wav

Context: Powerful background noise.

**Ground Truth**:

```
i got my book say say where's the badguy saloon you just go straight through the western town take a right and a left and you see it right there thanks honey clears throat man 1 can we move to number two man 2 how is his bounce man 1 can i get a bounce there man 2 just grab the crescent wrench come right back just make it a quick one tim man 1 looks great right there
```

**Prediction**:

```
agag i'm a buy say say where's a bad gods alone he just go straight to the western town take a right and a left and you see it right there aks hunny  an a u  cure owe  like a ai er
```

#### segment_0000001.wav

Context: Two male speakers, background music.

**Ground Truth**:

```
about to get his jaw busted grunts amateurs try and take men in alive all grunt amateurs usually don't make it announcer whether you're dead or alive you're just a dollar sign to jake cahill on bounty law thursdays at 830 only on nbc nbc theme plays hello everybody this is allen kincade on the set of the exciting hit nbc and screen gems television series bounty law
```

**Prediction**:

```
it is jumus o amateur's tryind take men in alive  a amateurs usually don't make it whether you're dead or alive you're just do dollar sign to cake cake law n shoty law thursday surday thirty only on n b c hellow everybody tis as allan chin kade on the set of the exciting hit n b c and screen gem stellivision series bouty law
```