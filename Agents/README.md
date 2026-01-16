# AI Agent System

A comprehensive AI agent system built with Streamlit and OpenAI GPT-4, featuring 6 specialized agents that work together to help you build applications from concept to deployment.

## 🤖 Agents Overview

### 1. **Orchestrator Agent** 🎯
- **Role**: Orchestrates the entire development process
- **Features**:
  - Project planning and timeline generation
  - Resource planning and risk assessment
  - Coordinates between all other agents
  - Tracks project progress and status
  - Provides next steps and recommendations

### 2. **Specifications Agent** 📋
- **Role**: Analyzes business requirements and creates specifications
- **Features**:
  - Detailed requirement analysis
  - User story generation
  - Acceptance criteria creation
  - Business rule identification
  - Stakeholder analysis
  - Impact analysis

### 3. **System Analyst Agent** 🏗️
- **Role**: Designs system architecture and integration patterns
- **Features**:
  - System architecture design
  - API design and documentation
  - Database schema design
  - Integration planning
  - Technology stack recommendations
  - Security analysis

### 4. **Coding Agent** 💻
- **Role**: Generates clean, efficient code
- **Features**:
  - Code generation for multiple languages
  - Framework-specific implementations
  - Code optimization
  - Bug fixing
  - Documentation generation
  - Template creation

### 5. **Code Reviewer Agent** 🔍
- **Role**: Reviews code for quality, security, and best practices
- **Features**:
  - Comprehensive code reviews
  - Security vulnerability detection
  - Performance analysis
  - Documentation quality review
  - Best practice enforcement
  - Quality metrics

### 6. **Tester Agent** 🧪
- **Role**: Creates and executes comprehensive tests
- **Features**:
  - Test case generation
  - Test plan creation
  - Bug report generation
  - Test execution simulation
  - Coverage analysis
  - Quality metrics

### 7. **Deployment Agent** 🚀
- **Role**: Assists in deploying applications to web services
- **Features**:
  - Multi-platform deployment support (Streamlit Cloud, Docker, Heroku)
  - Deployment package creation and validation
  - Platform-specific configuration generation
  - Deployment command generation
  - Performance and security analysis
  - AI-powered deployment assistance

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- OpenAI API key

### Installation

1. **Clone or download the project**
   ```bash
   cd /path/to/your/project
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**
   Create a `.env` file in the project root:
   ```bash
   OPENAI_API_KEY=your_openai_api_key_here
   ```

4. **Run the main application**
   ```bash
   streamlit run main.py
   ```

## 📁 Project Structure

```
AI-Agent-System/
├── main.py                 # Main orchestrator application
├── config.py              # Configuration settings
├── requirements.txt       # Python dependencies
├── README.md             # This file
├── utils/
│   ├── __init__.py
│   └── openai_client.py  # OpenAI API client
└── agents/
    ├── __init__.py
    ├── orchestrator.py       # Orchestrator Agent
    ├── specifications_agent.py   # Specifications Agent
    ├── system_analyst.py     # System Analyst Agent
    ├── coding_agent.py       # Coding Agent
    ├── code_reviewer.py      # Code Reviewer Agent
    ├── tester.py            # Tester Agent
    └── deployment_agent.py   # Deployment Agent
```

## 🎯 Usage

### Option 1: Orchestrated Workflow (Recommended)
1. Launch the **Orchestrator Agent**
2. Enter your project description
3. Follow the guided workflow through all phases
4. The Orchestrator will coordinate all other agents automatically

### Option 2: Individual Agent Usage
1. Launch the main application: `streamlit run main.py`
2. Choose which agent to use based on your current needs
3. Each agent can be used independently for specific tasks

### Workflow Example
```
Project Description → Business Analysis → System Design → Code Generation → Code Review → Testing → Deployment
```

## 🔧 Configuration

### OpenAI API Key
- Get your API key from [OpenAI Platform](https://platform.openai.com/)
- Add it to your `.env` file as `OPENAI_API_KEY=your_key_here`

### Model Configuration
- All agents use GPT-4 by default
- Models can be configured in `config.py`
- Temperature settings can be adjusted in each agent's interface

## 🛠️ Features

### Each Agent Includes:
- **Interactive Streamlit Interface**: User-friendly web interface
- **OpenAI Integration**: Powered by GPT-4 for intelligent responses
- **Download Capabilities**: Export generated content as files
- **Customizable Parameters**: Adjust temperature, focus areas, etc.
- **Specialized Tools**: Agent-specific utilities and templates

### System Features:
- **Modular Design**: Each agent is independent and can be used separately
- **Orchestration**: Orchestrator coordinates the entire workflow
- **Progress Tracking**: Visual progress indicators and status updates
- **Quality Assurance**: Built-in review and testing capabilities

## 📊 Agent Capabilities

| Agent | Input | Output | Key Features |
|-------|-------|--------|--------------|
| Orchestrator | Project description | Complete project plan | Orchestration, timeline, resources |
| Specifications Agent | Requirements | Detailed specifications | User stories, acceptance criteria |
| System Analyst | Specifications | Architecture design | APIs, databases, integrations |
| Coding Agent | Requirements + Design | Production code | Multiple languages, frameworks |
| Code Reviewer | Code | Review report | Security, performance, quality |
| Tester | Code | Test suite | Unit, integration, E2E tests |

## 🔒 Security & Best Practices

- **API Key Security**: Store OpenAI API keys in environment variables
- **Code Quality**: All agents follow best practices and coding standards
- **Error Handling**: Comprehensive error handling and user feedback
- **Input Validation**: Proper validation of user inputs
- **Documentation**: Extensive documentation and code comments

## 🚨 Troubleshooting

### Common Issues:

1. **OpenAI API Key Error**
   - Ensure your API key is correctly set in the `.env` file
   - Verify the key has sufficient credits

2. **Import Errors**
   - Make sure all dependencies are installed: `pip install -r requirements.txt`
   - Check Python version compatibility

3. **Agent Launch Issues**
   - Ensure all agent files are present in the `agents/` directory
   - Check file permissions

4. **Streamlit Issues**
   - Try refreshing the browser
   - Check the terminal for error messages

## 🤝 Contributing

This is a demonstration project showcasing AI agent orchestration. Feel free to:
- Extend agent capabilities
- Add new agents
- Improve the user interface
- Add new integrations

## 📄 License

This project is for educational and demonstration purposes. Please ensure you comply with OpenAI's usage policies when using their API.

## 🆘 Support

For issues or questions:
1. Check the troubleshooting section above
2. Verify your OpenAI API key and credits
3. Ensure all dependencies are properly installed
4. Check the Streamlit documentation for UI-related issues

---

**Built with ❤️ using Streamlit and OpenAI GPT-4**
