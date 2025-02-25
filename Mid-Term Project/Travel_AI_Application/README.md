# Travel Assistant AI

## Overview

Travel Assistant AI is a sophisticated multi-agent system designed to provide comprehensive travel assistance through a conversational interface. The application uses a specialized agent architecture to handle different aspects of travel planning, including itinerary creation, flight information, accommodation recommendations, and general travel advice.

## Features

- **Multi-Agent System**: Uses specialized agents for different travel needs:
  - **Itinerary Agent**: Creates customized travel itineraries with day-by-day planning
  - **Flight Agent**: Provides flight information, comparisons, and booking strategies
  - **Accommodation Agent**: Recommends hotels and accommodations based on user preferences
  - **Information Agent**: Offers general travel information and advice about destinations

- **Retrieval-Augmented Generation (RAG)**: Enhances responses with information from a dedicated travel knowledge base

- **Interactive Chat Interface**: Clean, intuitive Streamlit interface for natural conversation

- **Contextual Understanding**: Extracts and remembers relevant parameters from user queries

## Technical Architecture

The system is built on a modular architecture with the following components:

1. **Router Agent**: Analyzes user queries and directs them to the appropriate specialized agent
2. **Specialized Agents**: Domain-specific agents for itineraries, flights, accommodations, and information
3. **Knowledge Base**: FAISS vector database with travel information for accurate responses
4. **Response Generator**: Creates well-structured, informative replies with appropriate formatting

## Technologies Used

- **Streamlit**: For the interactive web interface
- **LangChain**: For agent coordination and RAG implementation
- **OpenAI GPT-4**: Powering the natural language understanding and generation
- **FAISS**: For efficient vector similarity search in the knowledge base
- **Python**: Core programming language

## Installation

### Prerequisites
- Python 3.8+
- OpenAI API key

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/travel-assistant.git
   cd travel-assistant
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up environment variables:
   ```bash
   export OPENAI_API_KEY=your_api_key_here
   ```
   Or create a `.env` file with:
   ```
   OPENAI_API_KEY=your_api_key_here
   ```

4. Run the application:
   ```bash
   streamlit run app_streamlit.py
   ```

## Usage

1. Launch the application using the command above
2. Type your travel-related questions in the chat input
3. The system will automatically route your query to the appropriate specialized agent
4. Receive comprehensive, well-formatted responses to your travel queries

### Example Queries

- "Create a 5-day itinerary for Paris focusing on art and cuisine with a mid-range budget"
- "What are the best flight options from New York to Tokyo in July?"
- "Recommend family-friendly accommodations in Barcelona near the beach"
- "What's the best time to visit Bali and what should I know about local customs?"

## Deployment

The application can be deployed to Hugging Face Spaces:

1. Create a new Space on Hugging Face with the Streamlit SDK
2. Connect your GitHub repository
3. Add your OpenAI API key as a secret in the Space settings
4. The Space will automatically build and deploy your application

## Future Enhancements

- Integration with real-time flight and hotel booking APIs
- User account system for saving and managing travel plans
- Mobile-responsive design optimization
- Support for additional languages
- Offline mode with cached travel information

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- OpenAI for the powerful language models
- LangChain for the agent framework
- The Streamlit team for the interactive web framework
- Everyone who contributed to the travel information database