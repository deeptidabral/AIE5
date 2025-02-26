# Building an AI Travel Assistant with LangGraph, RAG and Multi-Agent Framework

## Introduction

This notebook demonstrates how to build an AI travel assistant that can:
1. Generate customized travel itineraries
2. Answer travel-related questions
3. Provide flight and hotel recommendations
4. Offer destination information using RAG

We'll implement a multi-agent architecture using LangGraph, integrate with Amadeus API for real travel data, and use RAG (Retrieval Augmented Generation) with vector embeddings to enhance responses with travel knowledge.

## Setup and Dependencies

```python
# Install required packages
!pip install langchain langchain-openai langgraph langchain_community langsmith 
!pip install chromadb openai tiktoken amadeus tqdm
!pip install pandas matplotlib seaborn plotly
```

```python
import os
import json
import requests
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import pandas as pd

# LangChain and LangGraph imports
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# LangGraph imports
from langgraph.graph import END, StateGraph

# Amadeus API client
from amadeus import Client, ResponseError
```

## Configuration

```python
# API Keys and Configuration
os.environ["OPENAI_API_KEY"] = "your-openai-api-key"

# Amadeus API credentials
AMADEUS_CLIENT_ID = "your-amadeus-client-id"
AMADEUS_CLIENT_SECRET = "your-amadeus-client-secret"

# LLM Configuration
llm = ChatOpenAI(model="gpt-4-turbo", temperature=0.7)
embeddings = OpenAIEmbeddings()

# Initialize Amadeus client
amadeus = Client(
    client_id=AMADEUS_CLIENT_ID,
    client_secret=AMADEUS_CLIENT_SECRET
)
```

## Step 1: Building the Knowledge Base with Amadeus API Data

First, let's collect travel data from Amadeus API to build our retrieval database. We'll focus on:
1. Popular destinations
2. Flight information
3. Hotel information
4. Travel advisories

```python
def get_amadeus_access_token():
    """Get Amadeus API access token."""
    auth_url = "https://test.api.amadeus.com/v1/security/oauth2/token"
    headers = {"Content-Type": "application/x-www-form-urlencoded"}
    data = {
        "grant_type": "client_credentials",
        "client_id": AMADEUS_CLIENT_ID,
        "client_secret": AMADEUS_CLIENT_SECRET
    }
    
    response = requests.post(auth_url, headers=headers, data=data)
    if response.status_code == 200:
        return response.json().get("access_token")
    else:
        print(f"Failed to get access token: {response.json()}")
        return None
```

### Fetch Popular Destinations Data

```python
def fetch_popular_destinations():
    """Fetch popular travel destinations using Amadeus API."""
    try:
        response = amadeus.shopping.flight_destinations.get(
            origin='NYC',
            maxPrice=500
        )
        destinations = response.data
        
        # Transform into documents for our knowledge base
        documents = []
        for dest in destinations:
            content = f"""
            Destination: {dest['destination']}
            Departure Date: {dest.get('departureDate', 'N/A')}
            Return Date: {dest.get('returnDate', 'N/A')}
            Price: {dest.get('price', {}).get('total', 'N/A')} {dest.get('price', {}).get('currency', 'USD')}
            """
            doc = Document(
                page_content=content,
                metadata={
                    "type": "popular_destination",
                    "destination_code": dest['destination'],
                    "price": dest.get('price', {}).get('total', 'N/A')
                }
            )
            documents.append(doc)
        return documents
    except ResponseError as error:
        print(f"Error fetching popular destinations: {error}")
        return []
```

### Fetch Hotel Information

```python
def fetch_hotel_information(city_code):
    """Fetch hotel information for a specific city."""
    try:
        response = amadeus.shopping.hotel_offers.get(
            cityCode=city_code
        )
        hotels = response.data
        
        # Transform into documents
        documents = []
        for hotel in hotels:
            hotel_info = hotel.get('hotel', {})
            offers = hotel.get('offers', [])
            price_info = offers[0].get('price', {}) if offers else {}
            
            content = f"""
            Hotel: {hotel_info.get('name', 'N/A')}
            Location: {city_code}
            Rating: {hotel_info.get('rating', 'N/A')}
            Price: {price_info.get('total', 'N/A')} {price_info.get('currency', 'USD')}
            Description: {hotel_info.get('description', {}).get('text', 'No description available')}
            Amenities: {', '.join(hotel_info.get('amenities', ['N/A']))}
            """
            
            doc = Document(
                page_content=content,
                metadata={
                    "type": "hotel_information",
                    "hotel_id": hotel_info.get('hotelId', 'unknown'),
                    "city_code": city_code,
                    "rating": hotel_info.get('rating', 'N/A')
                }
            )
            documents.append(doc)
        return documents
    except ResponseError as error:
        print(f"Error fetching hotel information: {error}")
        return []
```

### Fetch Flight Offers

```python
def fetch_flight_offers(origin, destination, departure_date):
    """Fetch flight offers between two destinations."""
    try:
        response = amadeus.shopping.flight_offers_search.get(
            originLocationCode=origin,
            destinationLocationCode=destination,
            departureDate=departure_date,
            adults=1,
            max=10
        )
        flights = response.data
        
        # Transform into documents
        documents = []
        for flight in flights:
            # Extract key information
            itineraries = flight.get('itineraries', [])
            price_info = flight.get('price', {})
            
            segments_info = []
            for itinerary in itineraries:
                for segment in itinerary.get('segments', []):
                    departure = segment.get('departure', {})
                    arrival = segment.get('arrival', {})
                    carrier = segment.get('carrierCode', 'Unknown')
                    
                    segment_info = f"""
                    Flight: {carrier} {segment.get('number', 'N/A')}
                    From: {departure.get('iataCode', 'N/A')} at {departure.get('at', 'N/A')}
                    To: {arrival.get('iataCode', 'N/A')} at {arrival.get('at', 'N/A')}
                    """
                    segments_info.append(segment_info)
            
            content = f"""
            Route: {origin} to {destination}
            Departure Date: {departure_date}
            Price: {price_info.get('total', 'N/A')} {price_info.get('currency', 'EUR')}
            Flight Details:
            {"".join(segments_info)}
            """
            
            doc = Document(
                page_content=content,
                metadata={
                    "type": "flight_offer",
                    "origin": origin,
                    "destination": destination,
                    "departure_date": departure_date,
                    "price": price_info.get('total', 'N/A')
                }
            )
            documents.append(doc)
        return documents
    except ResponseError as error:
        print(f"Error fetching flight offers: {error}")
        return []
```

### Fetch Travel Advisories

```python
def fetch_travel_advisories(country_code):
    """Fetch travel advisories for a specific country."""
    try:
        response = amadeus.safety.safety_rated_locations.get(
            safetyRatedLocationType='COUNTRY',
            countryCode=country_code
        )
        advisories = response.data
        
        documents = []
        for advisory in advisories:
            safety_scores = advisory.get('safetyScores', {})
            
            content = f"""
            Country: {country_code}
            Overall Safety Score: {safety_scores.get('overall', 'N/A')}
            Physical Harm Risk: {safety_scores.get('physicalHarm', 'N/A')}
            Theft Risk: {safety_scores.get('theft', 'N/A')}
            Political Unrest Risk: {safety_scores.get('politicalFreedom', 'N/A')}
            Health Risk: {safety_scores.get('health', 'N/A')}
            Last Updated: {advisory.get('updatedDateTime', 'N/A')}
            """
            
            doc = Document(
                page_content=content,
                metadata={
                    "type": "travel_advisory",
                    "country_code": country_code,
                    "overall_safety": safety_scores.get('overall', 'N/A')
                }
            )
            documents.append(doc)
        return documents
    except ResponseError as error:
        print(f"Error fetching travel advisories: {error}")
        return []
```

### Collecting and Preparing Static Travel Information

Let's also include static travel information for our knowledge base:

```python
def prepare_destination_info():
    """Prepare static destination information."""
    destinations = [
        {
            "city": "Paris",
            "country": "France",
            "description": "Known as the City of Light, Paris is famous for the Eiffel Tower, Louvre Museum, and exquisite cuisine. Best time to visit is April-June or September-October for mild weather and fewer crowds.",
            "attractions": ["Eiffel Tower", "Louvre Museum", "Notre-Dame Cathedral", "Montmartre", "Champs-Élysées"],
            "cuisine": ["Croissants", "Escargot", "Coq au Vin", "Macarons", "French Wine"],
            "transportation": "Excellent public transportation with Metro, buses and RER trains. The Paris Visite travel pass offers unlimited travel on all transport networks.",
            "weather": "Temperate climate with mild winters and warm summers. Spring (March-May) and Fall (September-November) are particularly pleasant."
        },
        {
            "city": "Tokyo",
            "country": "Japan",
            "description": "Tokyo is a fascinating blend of ultramodern and traditional, with neon-lit skyscrapers coexisting with historic temples. Best time to visit is March-April for cherry blossoms or October-November for autumn colors.",
            "attractions": ["Tokyo Skytree", "Senso-ji Temple", "Meiji Shrine", "Shibuya Crossing", "Imperial Palace"],
            "cuisine": ["Sushi", "Ramen", "Tempura", "Yakitori", "Matcha desserts"],
            "transportation": "Highly efficient train and subway system. The Japan Rail Pass can be cost-effective for travelers. Taxis are clean but expensive.",
            "weather": "Four distinct seasons, with hot humid summers and cold winters. Spring and autumn are the most comfortable seasons."
        },
        # Add more destinations as needed
    ]
    
    documents = []
    for dest in destinations:
        content = f"""
        Destination: {dest['city']}, {dest['country']}
        
        Description: {dest['description']}
        
        Top Attractions: {', '.join(dest['attractions'])}
        
        Local Cuisine: {', '.join(dest['cuisine'])}
        
        Transportation: {dest['transportation']}
        
        Weather: {dest['weather']}
        """
        
        doc = Document(
            page_content=content,
            metadata={
                "type": "destination_info",
                "city": dest['city'],
                "country": dest['country']
            }
        )
        documents.append(doc)
    
    return documents
```

### Building the Vector Store

Now, let's build our vector store by combining all this travel information:

```python
def build_travel_knowledge_base():
    """Build the complete travel knowledge base."""
    all_documents = []
    
    # Collect static destination information
    all_documents.extend(prepare_destination_info())
    
    # Collect dynamic information from Amadeus API
    # Popular destinations from NYC
    all_documents.extend(fetch_popular_destinations())
    
    # Sample hotel information for major cities
    for city_code in ['PAR', 'LON', 'NYC', 'TYO', 'ROM']:
        all_documents.extend(fetch_hotel_information(city_code))
    
    # Sample flight offers for popular routes
    today = datetime.now().strftime('%Y-%m-%d')
    next_month = (datetime.now() + timedelta(days=30)).strftime('%Y-%m-%d')
    
    popular_routes = [
        ('NYC', 'LON'), ('NYC', 'PAR'), ('NYC', 'ROM'),
        ('LON', 'PAR'), ('LON', 'ROM'), ('PAR', 'ROM')
    ]
    
    for origin, destination in popular_routes:
        all_documents.extend(fetch_flight_offers(origin, destination, next_month))
    
    # Sample travel advisories
    for country_code in ['FR', 'GB', 'US', 'JP', 'IT']:
        all_documents.extend(fetch_travel_advisories(country_code))
    
    # Split documents for better embedding
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100
    )
    split_documents = text_splitter.split_documents(all_documents)
    
    # Create vector store
    vector_store = Chroma.from_documents(
        documents=split_documents,
        embedding=embeddings,
        collection_name="travel_knowledge_base"
    )
    
    return vector_store
```

```python
# Build the knowledge base
travel_db = build_travel_knowledge_base()
print(f"Knowledge base built with {travel_db._collection.count()} documents")
```

## Step 2: Building the Multi-Agent System with LangGraph

Now, let's build our multi-agent system using LangGraph. We'll create several specialized agents:

1. **Router Agent**: Determines which specialized agent should handle the query
2. **Itinerary Agent**: Creates customized travel itineraries
3. **Flight Agent**: Handles flight-related questions and searches
4. **Accommodation Agent**: Provides hotel recommendations
5. **Information Agent**: Answers general travel questions using RAG

### 1. Define Agent States and Workflows

```python
# Define the state for our agent system
class AgentState:
    def __init__(
        self,
        query: str,
        chat_history: Optional[List] = None,
        agent_executor: Optional[str] = None,
        agent_response: Optional[str] = None,
        final_response: Optional[str] = None,
        context: Optional[Dict] = None,
        error: Optional[str] = None
    ):
        self.query = query
        self.chat_history = chat_history or []
        self.agent_executor = agent_executor
        self.agent_response = agent_response
        self.final_response = final_response
        self.context = context or {}
        self.error = error
        
    def __repr__(self):
        return f"AgentState(query={self.query}, agent_executor={self.agent_executor})"
```

### 2. Router Agent - Determines which specialized agent to use

```python
def router_agent(state: AgentState) -> AgentState:
    """Router agent that determines which specialized agent should handle the query."""
    router_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a travel assistant router. Your job is to determine which specialized agent
        should handle the user's travel-related query. Choose the most appropriate agent from:
        
        - itinerary_agent: For requests to create travel itineraries, vacation plans, or multi-day travel schedules
        - flight_agent: For questions about flights, airfares, airlines, or flight bookings
        - accommodation_agent: For questions about hotels, accommodations, or places to stay
        - information_agent: For general travel information, destination facts, or travel advice
        
        Respond ONLY with the name of the appropriate agent. Do not include any explanations or additional text.
        """),
        ("human", "{query}")
    ])
    
    chain = router_prompt | llm | StrOutputParser()
    agent_executor = chain.invoke({"query": state.query}).strip()
    
    valid_agents = ["itinerary_agent", "flight_agent", "accommodation_agent", "information_agent"]
    if agent_executor not in valid_agents:
        # Default to information agent if invalid response
        agent_executor = "information_agent"
        
    # Update state with selected agent
    state.agent_executor = agent_executor
    return state
```

### 3. RAG Chain Setup - For Information Retrieval

```python
def setup_rag_chain():
    """Set up the RAG chain for information retrieval."""
    # Create a retriever from our travel knowledge base
    retriever = travel_db.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 5}
    )
    
    # Create the RAG prompt
    rag_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a knowledgeable travel assistant with expertise in destinations worldwide.
        Use the following travel information to provide detailed, accurate responses to the user's query.
        If the retrieved information doesn't fully answer the question, use your knowledge to provide
        the best possible response, but prioritize the retrieved information.
        
        Retrieved information:
        {context}
        """),
        ("human", "{query}")
    ])
    
    # Create the document chain
    document_chain = create_stuff_documents_chain(rag_prompt, llm)
    
    # Create and return the retrieval chain
    rag_chain = create_retrieval_chain(retriever, document_chain)
    return rag_chain
```

### 4. Specialized Agents Implementation

```python
def itinerary_agent(state: AgentState) -> AgentState:
    """Creates customized travel itineraries based on user preferences."""
    # First, retrieve relevant destination information
    rag_chain = setup_rag_chain()
    retrieval_result = rag_chain.invoke({"query": state.query})
    
    # Extract destination information from query using LLM
    extraction_prompt = ChatPromptTemplate.from_messages([
        ("system", """Extract the key travel information from the user's query.
        Return a JSON object with these fields (leave empty if not mentioned):
        {
            "destinations": ["list of mentioned destinations"],
            "duration": "total trip duration in days",
            "budget": "budget information",
            "interests": ["list of mentioned interests/activities"],
            "travel_dates": "approximate travel dates",
            "travelers": "number and type of travelers (family, couple, solo, etc.)"
        }
        """),
        ("human", "{query}")
    ])
    
    extraction_chain = extraction_prompt | llm | StrOutputParser()
    try:
        extracted_info = json.loads(extraction_chain.invoke({"query": state.query}))
        # Update context with extracted travel parameters
        state.context.update(extracted_info)
    except json.JSONDecodeError:
        # Handle parsing errors gracefully
        pass
    
    # Generate itinerary using retrieved information and extracted parameters
    itinerary_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a travel itinerary expert. Create a detailed day-by-day travel itinerary
        based on the user's preferences and the retrieved destination information.
        
        For each day, include:
        1. Morning activities
        2. Lunch suggestions
        3. Afternoon activities
        4. Dinner recommendations
        5. Evening activities or relaxation options
        
        Also include practical advice like:
        - Transportation between attractions
        - Estimated costs
        - Time management tips
        - Local customs to be aware of
        
        Make the itinerary realistic in terms of travel times and activities per day.
        
        Context information:
        {context_str}
        
        Extracted travel parameters:
        {parameters}
        """),
        ("human", "{query}")
    ])
    
    # Format the context and parameters for the prompt
    context_str = retrieval_result.get("context", "")
    parameters_str = json.dumps(state.context, indent=2)
    
    itinerary_chain = itinerary_prompt | llm | StrOutputParser()
    response = itinerary_chain.invoke({
        "query": state.query,
        "context_str": context_str,
        "parameters": parameters_str
    })
    
    state.agent_response = response
    return state
```

```python
def flight_agent(state: AgentState) -> AgentState:
    """Handles flight-related questions and searches."""
    # Extract flight search parameters
    extraction_prompt = ChatPromptTemplate.from_messages([
        ("system", """Extract flight search parameters from the user's query.
        Return a JSON object with these fields (leave empty if not mentioned):
        {
            "origin": "origin airport or city code",
            "destination": "destination airport or city code",
            "departure_date": "departure date in YYYY-MM-DD format",
            "return_date": "return date in YYYY-MM-DD format (if round-trip)",
            "num_passengers": "number of passengers",
            "cabin_class": "economy/business/first",
            "price_range": "budget constraints",
            "airline_preferences": ["preferred airlines"]
        }
        """),
        ("human", "{query}")
    ])
    
    extraction_chain = extraction_prompt | llm | StrOutputParser()
    try:
        flight_params = json.loads(extraction_chain.invoke({"query": state.query}))
        state.context.update({"flight_params": flight_params})
    except json.JSONDecodeError:
        # Handle parsing errors
        state.context.update({"flight_params": {}})
    
    # Get flight information using RAG
    rag_chain = setup_rag_chain()
    retrieval_result = rag_chain.invoke({"query": state.query})
    
    # Generate response based on flight parameters and retrieved information
    flight_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a flight search specialist. Provide helpful information about flights
        based on the retrieved flight data and the user's query. Include details about:
        
        - Available flights matching the criteria
        - Price ranges and fare comparisons
        - Airline options
        - Departure/arrival times
        - Travel duration
        - Layovers (if applicable)
        - Booking recommendations
        
        If exact flight information isn't available in the retrieved data, provide general advice
        about the requested route, typical prices, and best booking strategies.
        
        Retrieved flight information:
        {context}
        
        Extracted flight parameters:
        {flight_params}
        """),
        ("human", "{query}")
    ])
    
    flight_chain = flight_prompt | llm | StrOutputParser()
    response = flight_chain.invoke({
        "query": state.query,
        "context": retrieval_result.get("context", ""),
        "flight_params": json.dumps(state.context.get("flight_params", {}), indent=2)
    })
    
    state.agent_response = response
    return state
```

```python
def accommodation_agent(state: AgentState) -> AgentState:
    """Provides hotel and accommodation recommendations."""
    # Extract accommodation preferences
    extraction_prompt = ChatPromptTemplate.from_messages([
        ("system", """Extract accommodation preferences from the user's query.
        Return a JSON object with these fields (leave empty if not mentioned):
        {
            "location": "city or specific area",
            "check_in_date": "in YYYY-MM-DD format",
            "check_out_date": "in YYYY-MM-DD format",
            "guests": "number of guests",
            "rooms": "number of rooms",
            "budget_range": "price range per night",
            "amenities": ["list of desired amenities"],
            "property_type": "hotel/hostel/apartment/resort",
            "star_rating": "minimum star rating"
        }
        """),
        ("human", "{query}")
    ])
    
    extraction_chain = extraction_prompt | llm | StrOutputParser()
    try:
        accommodation_params = json.loads(extraction_chain.invoke({"query": state.query}))
        state.context.update({"accommodation_params": accommodation_params})
    except json.JSONDecodeError:
        state.context.update({"accommodation_params": {}})
    
    # Get accommodation information using RAG
    rag_chain = setup_rag_chain()
    retrieval_result = rag_chain.invoke({"query": f"hotels in {state.context.get('accommodation_params', {}).get('location', '')}"})
    
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
    
    accommodation_chain = accommodation_prompt | llm | StrOutputParser()
    response = accommodation_chain.invoke({
        "query": state.query,
        "context": retrieval_result.get("context", ""),
        "accommodation_params": json.dumps(state.context.get("accommodation_params", {}), indent=2)
    })
    
    state.agent_response = response
    return state
```

```python
def information_agent(state: AgentState) -> AgentState:
    """Answers general travel questions using RAG."""
    # This agent directly uses the RAG chain to provide travel information
    rag_chain = setup_rag_chain()
    result = rag_chain.invoke({"query": state.query})
    
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
        ("human", "{query}")
    ])
    
    enhancement_chain = enhancement_prompt | llm | StrOutputParser()
    enhanced_response = enhancement_chain.invoke({
        "query": state.query,
        "rag_response": result.get("answer", "")
    })
    
    state.agent_response = enhanced_response
    return state
```

### 5. Response Generator - Creates the final response

```python
def generate_final_response(state: AgentState) -> AgentState:
    """Generates the final, polished response to the user."""
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
    
    formatting_chain = formatting_prompt | llm | StrOutputParser()
    final_response = formatting_chain.invoke({
        "query": state.query,
        "agent_response": state.agent_response
    })
    
    state.final_response = final_response
    return state
```

### 6. Error Handler - Manages errors gracefully

```python
def handle_error(state: AgentState) -> AgentState:
    """Handles errors and provides a graceful fallback response."""
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
    
    error_chain = error_prompt | llm | StrOutputParser()
    fallback_response = error_chain.invoke({
        "query": state.query,
        "error": state.error or "Unknown error occurred"
    })
    
    state.final_response = fallback_response
    return state
```

## Step 3: Building the LangGraph Workflow

Now, let's connect all our agents into a workflow using LangGraph:

```python
def create_travel_assistant_graph():
    """Creates the travel assistant graph using LangGraph."""
    # Initialize the workflow graph
    workflow = StateGraph(AgentState)
    
    # Add nodes for each agent and processing step
    workflow.add_node("router", router_agent)
    workflow.add_node("itinerary_agent", itinerary_agent)
    workflow.add_node("flight_agent", flight_agent)
    workflow.add_node("accommodation_agent", accommodation_agent)
    workflow.add_node("information_agent", information_agent)
    workflow.add_node("response_generator", generate_final_response)
    workflow.add_node("error_handler", handle_error)
    
    # Define the graph edges and logic
    # Start with the router
    workflow.set_entry_point("router")
    
    # Connect router to specialized agents
    workflow.add_conditional_edges(
        "router",
        lambda state: state.agent_executor,
        {
            "itinerary_agent": lambda state: state.agent_executor == "itinerary_agent",
            "flight_agent": lambda state: state.agent_executor == "flight_agent",