import os
import json
import pickle
import streamlit as st
from typing import List, Dict, Any, Optional

from langchain_openai import ChatOpenAI
from langchain.chains import create_retrieval_chain                        
from langchain.chains.combine_documents import create_stuff_documents_chain  
from langchain.prompts import ChatPromptTemplate
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings

# Define the AgentState class
class AgentState:
    """State for the travel assistant workflow."""

    def __init__(self, query, chat_history=None, agent_executor=None, agent_response=None, 
                 final_response=None, context=None, error=None, messages=None):
        self.query = query
        self.chat_history = chat_history or []
        self.agent_executor = agent_executor
        self.agent_response = agent_response
        self.final_response = final_response
        self.context = context or {}
        self.error = error
        self.messages = messages or []
    
    def __repr__(self):
        return f"AgentState(query={self.query}, agent_executor={self.agent_executor})"
        
    def to_dict(self):
        """Convert state to dictionary for compatibility"""
        return {
            "query": self.query,
            "chat_history": self.chat_history,
            "agent_executor": self.agent_executor,
            "agent_response": self.agent_response,
            "final_response": self.final_response,
            "context": self.context,
            "error": self.error,
            "messages": self.messages
        }
    




# Initialize the LLM
@st.cache_resource
def load_llm():
    return ChatOpenAI(model="gpt-4-turbo", temperature=0.7)

# Load knowledge base
@st.cache_resource
def load_knowledge_base():
    try:
        travel_db = FAISS.load_local("travel_db_faiss", OpenAIEmbeddings(), allow_dangerous_deserialization=True)
        
        # Load the docstore
        with open('travel_db_docstore.pkl', 'rb') as f:
            travel_db.docstore = pickle.load(f)
            
        st.sidebar.success(f"Loaded knowledge base with {len(travel_db.index_to_docstore_id)} documents")
        return travel_db
    except Exception as e:
        st.sidebar.error(f"Error loading knowledge base: {e}")
        st.sidebar.info("Continuing without knowledge base. Responses will be based on LLM knowledge only.")
        return None

# Define agent functions
def router_agent(state):
    """Router agent that determines which specialized agent should handle the query."""
    llm = load_llm()

    # Create a chat prompt template with system and human messages
    router_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a travel assistant router. Your job is to determine which specialized agent
        should handle the user's travel-related query. Choose the most appropriate agent from:
        
        - itinerary_agent: For requests to create travel itineraries, vacation plans, or multi-day travel schedules
        - flight_agent: For questions about flights, airfares, airlines, or flight bookings
        - accommodation_agent: For questions about hotels, resorts, accommodations, or places to stay
        - information_agent: For general travel information, destination facts, or travel advice
        
        Respond ONLY with the name of the appropriate agent. Do not include any explanations or additional text.
        """),
        ("human", "{query}")
    ])
    
    # Use a chain to pass the user query to the LLM, get the recommended agent name
    response = llm.invoke(router_prompt.format(query=state.query))
    agent_executor = response.content.strip()
    
    valid_agents = ["itinerary_agent", "flight_agent", "accommodation_agent", "information_agent"]

    # Validate the response
    if agent_executor not in valid_agents:
        # Default to information agent if invalid response
        agent_executor = "information_agent"
            
    # Return state with updated agent_executor
    state.agent_executor = agent_executor
    return state

# RAG chain setup for Information Retrieval
def setup_rag_chain():
    """Set up the RAG chain for information retrieval."""
    travel_db = load_knowledge_base()
    if travel_db is None:
        return None
        
    # Create a retriever from our travel knowledge base
    retriever = travel_db.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 7}
    )
            
    rag_prompt = ChatPromptTemplate.from_template("""You are a knowledgeable travel assistant with expertise in destinations worldwide.
        Use the following travel information to provide detailed, accurate responses to the user's query.
        If the retrieved information doesn't fully answer the question, use your knowledge to provide
        the best possible response, but prioritize the retrieved information.
        
        Retrieved information: {context}        
        Question: {input}
    """)

    # Create the document processing chain
    document_chain = create_stuff_documents_chain(load_llm(), rag_prompt)
    
    return create_retrieval_chain(retriever, document_chain)

# Import all your agent functions
def itinerary_agent(state):
    """Creates customized travel itineraries based on user preferences."""
    llm = load_llm()
    try:
        # First, retrieve relevant destination information
        rag_chain = setup_rag_chain()
        if rag_chain:
            retrieval_result = rag_chain.invoke({"input": state.query})
        else:
            retrieval_result = {"answer": "I don't have specific information about this destination, but I'll do my best to help."}
        
        # Extract destination information from query using LLM
        extraction_prompt = ChatPromptTemplate.from_template("""Extract the key travel information from the user's query.
        Return a JSON object with these fields (leave empty if not mentioned):
        {{
            "destinations": ["list of mentioned destinations"],
            "duration": "total trip duration in days",
            "budget": "budget information",
            "interests": ["list of mentioned interests/activities"],
            "travel_dates": "approximate travel dates",
            "travelers": "number and type of travelers (family, couple, solo, etc.)"
        }}
        
        Query: {input}
        """)       
        
        extraction_result = llm.invoke(extraction_prompt.format(input=state.query))

        try:
            # First check if the string is not empty
            if extraction_result.content.strip():
                extracted_info = json.loads(extraction_result.content)
            else:
                extracted_info = {
                    "destinations": [],
                    "duration": "",
                    "budget": "",
                    "interests": [],
                    "travel_dates": "",
                    "travelers": ""
                }
            state.context.update(extracted_info)
        except json.JSONDecodeError:
            # Provide default structure when parsing fails
            extracted_info = {
                "destinations": [],
                "duration": "",
                "budget": "",
                "interests": [],
                "travel_dates": "",
                "travelers": ""
            }
            state.context.update(extracted_info)

        
        # Generate itinerary using retrieved information and extracted parameters
        itinerary_prompt = ChatPromptTemplate.from_template("""You are a travel itinerary expert. Create a detailed day-by-day travel itinerary
        based on the user's preferences and the retrieved destination information.
        
        For each day, include morning activities, lunch suggestions, afternoon activities, dinner recommendations, and 
        evening activities or relaxation options
        
        Also include practical advice about transportation between attractions, estimated costs, time management tips, and local customs.
        
        Make the itinerary realistic in terms of travel times and activities per day.
        
        Context information:
        {context_str}
        
        Extracted travel parameters:
        {parameters}
        
        Query: {input}
        """)        
        
        # Format the context and parameters for the prompt
        context_str = retrieval_result.get("answer", "")
        parameters_str = json.dumps(state.context, indent=2)
        
        response = llm.invoke(itinerary_prompt.format(
            input=state.query,
            context_str=context_str,
            parameters=parameters_str
        ))
        
        state.agent_response = response.content
        return state
        
    except Exception as e:
        state.error = str(e)
        return state

# Include the other agent functions: flight_agent, accommodation_agent, information_agent, etc.
# ... (Copy them from your original code)

def flight_agent(state):
    """Handles flight-related questions and searches."""
    llm = load_llm()
    try:
        # Extract flight search parameters
        extraction_prompt = ChatPromptTemplate.from_template("""Extract flight search parameters from the user's query.
            Return a JSON object with these fields (leave empty if not mentioned):
        {{
            "origin": "origin airport or city code",
            "destination": "destination airport or city code",
            "departure_date": "departure date in YYYY-MM-DD format",
            "return_date": "return date in YYYY-MM-DD format (if round-trip)",
            "num_passengers": "number of passengers",
            "cabin_class": "economy/business/first",
            "price_range": "budget constraints",
            "airline_preferences": ["preferred airlines"]
        }}
        
        Query: {input}
        """)       
        
        extraction_result = llm.invoke(extraction_prompt.format(input=state.query))

        try:
            flight_params = json.loads(extraction_result.content)
            state.context.update({"flight_params": flight_params})
        except json.JSONDecodeError:
            state.context.update({"flight_params": {}})
        
        # Get flight information using RAG
        rag_chain = setup_rag_chain()
        if rag_chain:
            retrieval_result = rag_chain.invoke({"input": state.query})
        else:
            retrieval_result = {"answer": "I don't have specific flight information in my database, but I can provide general advice."}

        flight_prompt = ChatPromptTemplate.from_template("""You are a flight search specialist. Provide helpful information about flights
            based on the retrieved flight data and the user's query. Include details about available flights matching the criteria, 
            price ranges and fare comparisons, airline options, departure/arrival times, travel duration, layovers (if applicable), 
            and booking recommendations.
            
            If exact flight information isn't available in the retrieved data, provide general advice
            about the requested route, typical prices, and best booking strategies.
            
            Retrieved flight information:
            {context_str}
            
            Extracted flight parameters:
            {parameters}
        
            Query: {input}
            """)
        
        # Format the context and parameters for the prompt
        context_str = retrieval_result.get("answer", "")
        parameters_str = json.dumps(state.context.get("flight_params", {}), indent=2)
        
        response = llm.invoke(flight_prompt.format(
            input=state.query,
            context_str=context_str,
            parameters=parameters_str
        ))
        
        state.agent_response = response.content
        return state
        
    except Exception as e:
        state.error = str(e)
        return state

def accommodation_agent(state):
    """Provides hotel and accommodation recommendations."""
    llm = load_llm()
    try:
        # Extract accommodation preferences
        extraction_prompt = ChatPromptTemplate.from_messages([
            ("system", """Extract accommodation preferences from the user's query.
            Return a JSON object with these fields (leave empty if not mentioned):
            {{
                "location": "city or specific area",
                "check_in_date": "in YYYY-MM-DD format",
                "check_out_date": "in YYYY-MM-DD format",
                "guests": "number of guests",
                "rooms": "number of rooms",
                "budget_range": "price range per night"
            }}
            """),
            ("human", "{query}")
        ])
        
        extraction_result = llm.invoke(extraction_prompt.format(query=state.query))
        try:
            accommodation_params = json.loads(extraction_result.content)
            state.context.update({"accommodation_params": accommodation_params})
        except json.JSONDecodeError:
            state.context.update({"accommodation_params": {}})
        
        # Get accommodation information using RAG
        rag_chain = setup_rag_chain()
        location = state.context.get('accommodation_params', {}).get('location', '')
        if rag_chain and location:
            retrieval_result = rag_chain.invoke({"input": f"hotels in {location}"})
        else:
            retrieval_result = {"context": "I don't have specific accommodation information in my database, but I can provide general advice."}
        
        # Generate response based on accommodation parameters and retrieved information
        accommodation_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a hotel and accommodation expert. Provide detailed recommendations
            based on the user's preferences and the retrieved accommodation data. Include:
            
            - Suitable hotel/accommodation options
            - Price ranges and value considerations
            - Location benefits and proximity to attractions
            - Amenities and facilities
            - Guest ratings and reviews summary
            - Booking tips and optimal timing
            
            If specific accommodation data isn't available, provide general advice about
            accommodations in the requested location, typical options at different price points,
            and best areas to stay.
            
            Retrieved accommodation information:
            {context}
            
            Extracted accommodation parameters:
            {accommodation_params}
            """),
            ("human", "{query}")
        ])
        
        response = llm.invoke(accommodation_prompt.format(
            query=state.query,
            context=retrieval_result.get("context", ""),
            accommodation_params=json.dumps(state.context.get("accommodation_params", {}), indent=2)
        ))
        
        state.agent_response = response.content
        return state
        
    except Exception as e:
        state.error = str(e)
        return state

def information_agent(state):
    """Answers general travel questions using RAG."""
    llm = load_llm()
    try:
        # This agent directly uses the RAG chain to provide travel information
        rag_chain = setup_rag_chain()
        if rag_chain:
            result = rag_chain.invoke({"input": state.query})
            rag_response = result.get("answer", "")
        else:
            rag_response = "I don't have specific information about this in my travel database, but I'll provide general advice based on my knowledge."
        
        # Enhance RAG response with additional context if needed
        enhancement_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a knowledgeable travel information specialist. Review and enhance
            the retrieved information to provide a comprehensive, accurate response to the user's query.
            
            If the retrieved information is incomplete, add relevant details from your knowledge while
            clearly distinguishing between retrieved facts and general knowledge.
            
            Focus on providing practical, useful information that directly addresses the user's needs.
            Include cultural insights, traveler tips, and seasonal considerations when relevant.
            
            Retrieved information:
            {rag_response}
            """),
            ("human", "{input}")
        ])
        
        response = llm.invoke(enhancement_prompt.format(
            input=state.query,
            rag_response=rag_response
        ))
        
        state.agent_response = response.content
        return state
        
    except Exception as e:
        state.error = str(e)
        return state

# Response Generator - Creates the final response
def generate_final_response(state):
    """Generates the final, polished response to the user."""
    llm = load_llm()
    # Create a consistent, helpful response format
    formatting_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a friendly, helpful travel assistant. Format the specialized agent's response
        into a clear, well-structured, and engaging reply. Maintain all the factual information and advice
        while improving readability with:
        
        - A warm, conversational tone
        - Logical organization with headings where appropriate
        - Bullet points for lists
        - Bold text for important information
        - Emojis where appropriate (but not excessive)
        
        Make sure the response completely addresses the user's query. Add a brief, friendly closing
        that invites further questions.
        
        Original agent response:
        {agent_response}
        """),
        ("human", "{query}")
    ])
    
    response = llm.invoke(formatting_prompt.format(
        query=state.query,
        agent_response=state.agent_response
    ))
    
    state.agent_response = response.content
    return state

# Error Handler - Manages errors gracefully
def handle_error(state):
    """Handles errors and provides a graceful fallback response."""
    llm = load_llm()
    error_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful travel assistant. The system encountered an error while
        processing the user's query. Provide a helpful response that:
        
        1. Acknowledges the issue
        2. Offers general travel advice related to their query
        3. Suggests how they might rephrase their question for better results
        
        Error message: {error}
        """),
        ("human", "{query}")
    ])
    
    response = llm.invoke(error_prompt.format(
        query=state.query,
        error=state.error or "Unknown error occurred"
    ))
    
    state.agent_response = response.content
    return state

# Streamlit app
def main():
    st.set_page_config(
        page_title="Travel Assistant",
        page_icon="✈️",
        layout="wide"
    )
    
    st.sidebar.title("Travel Assistant")
    st.sidebar.info("This app uses a multi-agent system to help with travel planning, flight information, accommodations, and general travel advice.")
    
    # Load the LLM and knowledge base
    llm = load_llm()
    travel_db = load_knowledge_base()
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "👋 Hello! I'm your Travel Assistant. I can help with travel itineraries, flight information, accommodations, and general travel advice. How can I assist you today?"}
        ]
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask me about travel..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Display assistant response
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            message_placeholder.markdown("Thinking...")
            
            # Process the query with multi-agent system
            try:
                # Initialize the state with the user query
                state = AgentState(query=prompt)
                
                # Route the query to the appropriate agent
                with st.spinner("Routing your query..."):
                    state = router_agent(state)
                
                agent_type = state.agent_executor
                st.session_state["last_agent"] = agent_type
                
                # Process the query with the appropriate handler
                with st.spinner(f"Processing with {agent_type.replace('_', ' ').title()}..."):
                    if agent_type == "itinerary_agent":
                        state = itinerary_agent(state)
                    elif agent_type == "flight_agent":
                        state = flight_agent(state)
                    elif agent_type == "accommodation_agent":
                        state = accommodation_agent(state)
                    else:  # Default to information agent
                        state = information_agent(state)
                
                # Check for errors
                if state.error:
                    state = handle_error(state)
                else:
                    # Format the response
                    with st.spinner("Formatting response..."):
                        state = generate_final_response(state)
                
                # Display the response
                message_placeholder.markdown(state.agent_response)
                
                # Add assistant response to chat history
                st.session_state.messages.append({"role": "assistant", "content": state.agent_response})
                
            except Exception as e:
                error_message = f"An error occurred: {str(e)}"
                message_placeholder.markdown(error_message)
                st.session_state.messages.append({"role": "assistant", "content": error_message})

if __name__ == "__main__":
    main()