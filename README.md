# Podcast Deputy by Dr. Tristan Behrens

## Installation

First and foremost, I hope you are either on a Mac or Linux. If you are on Windows, you might want to consider using the Windows Subsystem for Linux (WSL) or a virtual machine. Note, that I have not tested the installation on Windows and I am not planning to do so.

You might consider using a virtual environment to run the code. I am using `conda` for this purpose. If you are not familiar with `conda`, you might want to check out the [official documentation](https://docs.conda.io/en/latest/). Note that you can also use `venv` or `virtualenv` if you prefer, or even work without a virtual environment.

```
conda create -n "gradio" python=3.10.13
conda activate gradio
```

Now, you can install the required packages. This is obligatory, as the code will not run without them:

```
pip install -r requirements.txt
```

Next, you have to set up playwright:

```
playwright install
```

Finally, you have to create the file `.env` in the root directory of the project. This file should contain the following content:

```
ELEVENLABS_API_KEY=<ElevenLabs API Key>
ANTHROPIC_API_KEY=<Anthropic API Key>
```

You will get the ElevenLabs API key here: https://elevenlabs.io/app/settings/api-keys

You will get the Anthropic API key here: https://console.anthropic.com/settings/keys

## Usage

To run the code, you can use the following command:

```
gradio aiupdate_run.py
```