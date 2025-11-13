# ChatGPT-like Clone with Streamlit

A simple ChatGPT-like interface built with Streamlit and OpenAI's API.

## 🚀 Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/saraweave/Health-Score---Streamlit.git
cd Health-Score---Streamlit
```

### 2. Install Dependencies
```bash
pip install streamlit openai
```

### 3. Set Up Your OpenAI API Key

#### Option A: Local Development (Recommended)
1. Copy the template: `cp .streamlit/secrets.toml.template .streamlit/secrets.toml`
2. Edit `.streamlit/secrets.toml` and add your OpenAI API key:
   ```toml
   OPENAI_API_KEY = "sk-your-actual-api-key-here"
   ```

#### Option B: Environment Variable
Set your API key as an environment variable:
```bash
export OPENAI_API_KEY="sk-your-actual-api-key-here"
```

### 4. Run the App
```bash
streamlit run streamlit_app.py
```

## 🔑 Getting an OpenAI API Key
1. Visit [OpenAI Platform](https://platform.openai.com/api-keys)
2. Sign up or log in
3. Create a new API key
4. Copy and use it in your configuration

## 🚀 Deployment

### Streamlit Cloud
1. Fork this repository
2. Go to [Streamlit Cloud](https://streamlit.io/cloud)
3. Connect your GitHub account
4. Deploy your fork
5. Add your `OPENAI_API_KEY` in the Streamlit Cloud secrets management

### Other Platforms
For other deployment platforms, set the `OPENAI_API_KEY` environment variable.

## 📁 Project Structure
```
├── streamlit_app.py           # Main application
├── .streamlit/
│   ├── secrets.toml.template  # Template for API key configuration
│   └── secrets.toml          # Your actual API key (git-ignored)
├── .gitignore                # Excludes secrets from version control
└── README.md                 # This file
```

## 🔒 Security Notes
- Never commit your actual API key to version control
- The `secrets.toml` file is git-ignored for your protection
- Use environment variables for production deployments