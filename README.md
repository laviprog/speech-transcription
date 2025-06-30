# 🎙 Speech Transcription API

Speech Transcription API is a RESTful service that processes audio input and converts speech into text using state-of-the-art speech recognition models. Ideal for building transcription tools, smart assistants, and voice-controlled applications.

![Tests and Linting](https://github.com/laviprog/speech-transcription/actions/workflows/workflow.yml/badge.svg)


## 🚀 Features

- 🎤 Transcribe audio to text (STT, speech-to-text)
- 🔐 Secure JWT-based authentication
- ⚡ FastAPI backend with async support
- 🐳 Dockerized for easy deployment (CPU & GPU)


## 🛠️ Getting Started

Follow the steps below to set up and run the Speech Transcription API using Docker (with optional GPU acceleration).

### 📦 Install Dependencies

You can use either uv (recommended for speed) or pip.

#### Using `uv`:
```bash
uv sync
```

#### Using `pip`:
1. Create a virtual environment:
    ```bash
    python -m venv .venv
    ```
2. Activate the virtual environment:
    ```bash
    source .venv/bin/activate  # Linux/macOS
    # .venv\Scripts\activate   # Windows
    ```
3. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

### ⚙️ Configure Environment Variables

Copy the example environment file and fill in the necessary values:

```bash
cp .env.example .env
```

Edit the `.env` file to set your environment variables. You can use the default values or customize them as needed.

### 🐳 Build and Run the Docker Container

#### Using CPU:
Start the Docker container with the following command:

```bash
docker-compose up --build
```
This command will build the Docker image and start the container.

#### Using GPU:

Set up the `docker-compose.yml` file to use GPU acceleration.

```bash
docker-compose up --build
```

This command will build the Docker image and start the container with GPU support.

Then, API will be available at `http://localhost:8000`.
Documentation will be available at `http://localhost:8000/docs`.
