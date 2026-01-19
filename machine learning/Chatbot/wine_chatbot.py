"""
Wine Sommelier FAQ Chatbot
==========================
An AI-powered wine sommelier chatbot that uses LangChain with Anthropic Claude API
and RAG (Retrieval Augmented Generation) to answer questions about red wine.

This chatbot:
1. Loads and analyzes the wine quality dataset
2. Creates a knowledge base with wine information and statistics
3. Uses ChromaDB for vector storage and retrieval
4. Leverages LangChain for prompt management and chain orchestration
5. Provides personalized wine recommendations based on data analysis

Author: Wine Sommelier Bot
"""

import os
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import json

# LangChain imports
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_classic.chains import ConversationalRetrievalChain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_classic.memory import ConversationBufferMemory
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.documents import Document
from langchain_anthropic import ChatAnthropic


@dataclass
class WineProfile:
    """Data class representing a wine profile with quality characteristics."""
    quality_level: str
    avg_alcohol: float
    avg_acidity: float
    avg_ph: float
    avg_sulphates: float
    characteristics: List[str]
    food_pairings: List[str]
    description: str


class WineDataAnalyzer:
    """
    Analyzes wine dataset and generates insights for the knowledge base.
    """
    
    def __init__(self, csv_path: str):
        """Initialize with path to wine quality CSV file."""
        self.df = pd.read_csv(csv_path)
        self.quality_profiles = self._create_quality_profiles()
        
    def _create_quality_profiles(self) -> Dict[str, WineProfile]:
        """Create detailed profiles for each quality level."""
        profiles = {}
        
        quality_descriptions = {
            3: ("Poor", "Entry-level wines with notable flaws", 
                ["Simple flavors", "High acidity", "Harsh tannins"],
                ["Casual cooking", "Sangria base"]),
            4: ("Below Average", "Basic wines suitable for everyday consumption",
                ["Light body", "Subtle fruit notes", "Short finish"],
                ["Pizza", "Burgers", "Casual meals"]),
            5: ("Average", "Decent quality wines for everyday enjoyment",
                ["Balanced structure", "Moderate tannins", "Pleasant fruit"],
                ["Pasta", "Grilled vegetables", "Light meats"]),
            6: ("Above Average", "Good quality wines with notable character",
                ["Rich fruit flavors", "Smooth tannins", "Medium body"],
                ["Roasted chicken", "Beef stew", "Hard cheeses"]),
            7: ("Good", "High-quality wines with complexity and depth",
                ["Complex aromas", "Elegant structure", "Long finish"],
                ["Filet mignon", "Lamb", "Aged cheeses"]),
            8: ("Excellent", "Premium wines with exceptional qualities",
                ["Intense complexity", "Velvety tannins", "Outstanding balance"],
                ["Prime rib", "Duck", "Gourmet dishes"])
        }
        
        for quality in self.df['quality'].unique():
            subset = self.df[self.df['quality'] == quality]
            level, desc, chars, pairings = quality_descriptions.get(
                quality, 
                ("Unknown", "No description available", [], [])
            )
            
            profiles[str(quality)] = WineProfile(
                quality_level=level,
                avg_alcohol=round(subset['alcohol'].mean(), 2),
                avg_acidity=round(subset['fixed acidity'].mean(), 2),
                avg_ph=round(subset['pH'].mean(), 2),
                avg_sulphates=round(subset['sulphates'].mean(), 2),
                characteristics=chars,
                food_pairings=pairings,
                description=desc
            )
            
        return profiles
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics about the wine dataset."""
        stats = {
            "total_wines": len(self.df),
            "quality_distribution": self.df['quality'].value_counts().to_dict(),
            "alcohol_range": {
                "min": round(self.df['alcohol'].min(), 2),
                "max": round(self.df['alcohol'].max(), 2),
                "mean": round(self.df['alcohol'].mean(), 2)
            },
            "ph_range": {
                "min": round(self.df['pH'].min(), 2),
                "max": round(self.df['pH'].max(), 2),
                "mean": round(self.df['pH'].mean(), 2)
            },
            "acidity_stats": {
                "fixed_acidity_mean": round(self.df['fixed acidity'].mean(), 2),
                "volatile_acidity_mean": round(self.df['volatile acidity'].mean(), 2),
                "citric_acid_mean": round(self.df['citric acid'].mean(), 2)
            }
        }
        return stats
    
    def get_quality_factors(self) -> Dict[str, float]:
        """Analyze correlations between features and quality."""
        correlations = {}
        for col in self.df.columns:
            if col != 'quality':
                correlations[col] = round(self.df[col].corr(self.df['quality']), 3)
        return dict(sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True))
    
    def get_wine_recommendations(self, preferences: Dict[str, Any]) -> pd.DataFrame:
        """Get wine recommendations based on user preferences."""
        filtered = self.df.copy()
        
        if 'min_quality' in preferences:
            filtered = filtered[filtered['quality'] >= preferences['min_quality']]
        if 'min_alcohol' in preferences:
            filtered = filtered[filtered['alcohol'] >= preferences['min_alcohol']]
        if 'max_alcohol' in preferences:
            filtered = filtered[filtered['alcohol'] <= preferences['max_alcohol']]
        if 'low_acidity' in preferences and preferences['low_acidity']:
            filtered = filtered[filtered['volatile acidity'] < filtered['volatile acidity'].median()]
            
        return filtered.head(10)


class WineKnowledgeBase:
    """
    Creates and manages the wine knowledge base for RAG.
    """
    
    def __init__(self, analyzer: WineDataAnalyzer):
        self.analyzer = analyzer
        self.documents = self._create_documents()
        
    def _create_documents(self) -> List[Document]:
        """Create document chunks for the knowledge base."""
        documents = []
        
        # Document 1: Dataset Overview
        stats = self.analyzer.get_statistics()
        overview = f"""
        # Red Wine Quality Dataset Overview
        
        This knowledge base contains information about {stats['total_wines']} red wine samples 
        from the Portuguese "Vinho Verde" wine region. The wines have been rated on a quality 
        scale from 0 to 10, with our samples ranging from quality 3 to 8.
        
        ## Quality Distribution
        The wines in our database are distributed as follows:
        - Quality 3 (Poor): {stats['quality_distribution'].get(3, 0)} wines
        - Quality 4 (Below Average): {stats['quality_distribution'].get(4, 0)} wines
        - Quality 5 (Average): {stats['quality_distribution'].get(5, 0)} wines
        - Quality 6 (Above Average): {stats['quality_distribution'].get(6, 0)} wines
        - Quality 7 (Good): {stats['quality_distribution'].get(7, 0)} wines
        - Quality 8 (Excellent): {stats['quality_distribution'].get(8, 0)} wines
        
        ## Alcohol Content Range
        - Minimum: {stats['alcohol_range']['min']}%
        - Maximum: {stats['alcohol_range']['max']}%
        - Average: {stats['alcohol_range']['mean']}%
        
        ## pH Levels
        - Minimum: {stats['ph_range']['min']}
        - Maximum: {stats['ph_range']['max']}
        - Average: {stats['ph_range']['mean']}
        """
        documents.append(Document(
            page_content=overview,
            metadata={"source": "overview", "topic": "dataset_statistics"}
        ))
        
        # Document 2: Wine Chemistry Explained
        chemistry = """
        # Wine Chemistry Guide
        
        ## Fixed Acidity
        Fixed acidity refers to the non-volatile acids in wine, primarily tartaric acid.
        These acids contribute to the wine's crispness and freshness. Higher fixed acidity
        generally correlates with wines that feel more refreshing on the palate.
        Typical range: 4.6 to 15.9 g/dm³
        
        ## Volatile Acidity
        Volatile acidity is the gaseous acids present in wine, mainly acetic acid.
        High levels can lead to an unpleasant vinegar taste. Lower volatile acidity
        is generally preferred and correlates with higher quality wines.
        Typical range: 0.12 to 1.58 g/dm³
        
        ## Citric Acid
        Citric acid adds freshness and flavor complexity to wines. It can enhance
        the fruitiness and is often found in small quantities. Wines with moderate
        citric acid tend to taste more vibrant.
        Typical range: 0 to 1 g/dm³
        
        ## Residual Sugar
        Residual sugar is the natural grape sugars remaining after fermentation.
        Most red wines in our dataset are dry, with low residual sugar levels.
        This affects the perceived sweetness of the wine.
        Typical range: 0.9 to 15.5 g/dm³
        
        ## Chlorides
        Chlorides represent the salt content in wine. Higher chloride levels
        can contribute to a salty taste. Moderate levels are preferred.
        Typical range: 0.012 to 0.611 g/dm³
        
        ## Sulfur Dioxide (Free and Total)
        Sulfur dioxide acts as an antimicrobial and antioxidant in wine.
        Free SO2 prevents microbial growth and oxidation. Total SO2 includes
        both free and bound forms. Proper levels are essential for wine preservation.
        Free SO2 range: 1 to 72 mg/dm³
        Total SO2 range: 6 to 289 mg/dm³
        
        ## Density
        Wine density is affected by sugar and alcohol content. Higher alcohol
        reduces density, while sugar increases it. Density helps identify
        the body and structure of the wine.
        Typical range: 0.990 to 1.004 g/cm³
        
        ## pH
        pH measures the acidity level on a scale of 0-14. Wine pH typically
        ranges from 3 to 4, with lower values indicating higher acidity.
        pH affects the wine's stability, color, and taste.
        Typical range: 2.74 to 4.01
        
        ## Sulphates
        Sulphates contribute to the sulfur dioxide levels and act as
        antimicrobial agents. Higher sulphate content positively correlates
        with wine quality in red wines.
        Typical range: 0.33 to 2.0 g/dm³
        
        ## Alcohol
        Alcohol content significantly impacts wine body, warmth, and preservation.
        Higher alcohol wines tend to have more body and complexity. In our dataset,
        alcohol shows the strongest positive correlation with quality.
        Typical range: 8.4% to 14.9%
        """
        documents.append(Document(
            page_content=chemistry,
            metadata={"source": "chemistry", "topic": "wine_components"}
        ))
        
        # Document 3: Quality Factors
        factors = self.analyzer.get_quality_factors()
        quality_factors = f"""
        # What Makes a High-Quality Red Wine?
        
        Based on our analysis of {stats['total_wines']} red wines, here are the factors
        that most influence wine quality, ranked by their correlation strength:
        
        ## Top Quality Indicators (Positive Correlation)
        1. **Alcohol Content** (correlation: {factors.get('alcohol', 'N/A')})
           Higher alcohol content strongly correlates with better quality ratings.
           Premium wines typically have 11-14% alcohol content.
        
        2. **Sulphates** (correlation: {factors.get('sulphates', 'N/A')})
           Wines with higher sulphate levels tend to receive better quality scores.
           Optimal range: 0.6-1.0 g/dm³
        
        3. **Citric Acid** (correlation: {factors.get('citric acid', 'N/A')})
           Moderate citric acid adds complexity and freshness.
        
        4. **Fixed Acidity** (correlation: {factors.get('fixed acidity', 'N/A')})
           Proper acidity balance contributes to wine structure.
        
        ## Negative Quality Indicators
        1. **Volatile Acidity** (correlation: {factors.get('volatile acidity', 'N/A')})
           High volatile acidity leads to vinegar-like off-flavors.
           Keep below 0.6 g/dm³ for quality wines.
        
        2. **Total Sulfur Dioxide** (correlation: {factors.get('total sulfur dioxide', 'N/A')})
           Excessive SO2 can mask fruit flavors and cause harshness.
        
        3. **Density** (correlation: {factors.get('density', 'N/A')})
           Higher density often indicates lower alcohol conversion.
        
        ## Key Takeaways for Wine Selection
        - Look for wines with alcohol content between 11-13%
        - Avoid wines with noticeable volatile acidity (vinegar smell)
        - Well-balanced sulphates indicate better preservation
        - Moderate citric acid adds pleasant freshness
        """
        documents.append(Document(
            page_content=quality_factors,
            metadata={"source": "quality_factors", "topic": "quality_indicators"}
        ))
        
        # Document 4-9: Quality Level Profiles
        for quality, profile in self.analyzer.quality_profiles.items():
            profile_doc = f"""
            # Quality {quality} Wine Profile - {profile.quality_level}
            
            ## Overview
            {profile.description}
            
            ## Typical Characteristics
            - Average Alcohol: {profile.avg_alcohol}%
            - Average Fixed Acidity: {profile.avg_acidity} g/dm³
            - Average pH: {profile.avg_ph}
            - Average Sulphates: {profile.avg_sulphates} g/dm³
            
            ## Tasting Notes
            {', '.join(profile.characteristics)}
            
            ## Food Pairing Suggestions
            These wines pair well with:
            {', '.join(profile.food_pairings)}
            
            ## When to Choose This Quality Level
            Quality {quality} wines are {"best for special occasions and fine dining" 
            if int(quality) >= 7 else "suitable for everyday enjoyment and casual meals"
            if int(quality) >= 5 else "best used for cooking or casual gatherings"}.
            """
            documents.append(Document(
                page_content=profile_doc,
                metadata={"source": f"quality_{quality}", "topic": "quality_profile", "quality": quality}
            ))
        
        # Document 10: Wine Tasting Guide
        tasting_guide = """
        # Red Wine Tasting Guide
        
        ## The Five S's of Wine Tasting
        
        ### 1. See (Visual Examination)
        Hold your glass against a white background. Examine:
        - **Color intensity**: Range from pale ruby to deep purple
        - **Clarity**: Should be clear, not cloudy
        - **Viscosity**: Look at the "legs" running down the glass
        
        For red wines, deeper color often indicates:
        - Fuller body
        - Higher tannin content
        - Potentially higher quality
        
        ### 2. Swirl
        Gently swirl the wine to:
        - Release aromatic compounds
        - Observe the legs (higher alcohol = slower legs)
        - Aerate the wine slightly
        
        ### 3. Smell (Olfactory Examination)
        Primary aromas from grapes:
        - Fruits: Cherry, plum, blackberry, raspberry
        - Flowers: Violet, rose
        
        Secondary aromas from fermentation:
        - Yeast, bread, butter
        
        Tertiary aromas from aging:
        - Vanilla, oak, tobacco, leather
        
        ### 4. Sip
        Take a small sip and let it coat your palate:
        - **Sweetness**: Detected at the tip of tongue
        - **Acidity**: Causes mouth watering on sides
        - **Tannins**: Felt as dryness/astringency
        - **Body**: Light, medium, or full
        - **Finish**: How long flavors linger
        
        ### 5. Savor
        Consider the overall impression:
        - Is it balanced?
        - Is it complex or simple?
        - Would you enjoy another glass?
        
        ## Quality Indicators When Tasting
        
        **Signs of High Quality:**
        - Complex, layered aromas
        - Well-integrated tannins
        - Long, pleasant finish
        - Harmonious balance
        - Distinct character
        
        **Warning Signs:**
        - Vinegar smell (high volatile acidity)
        - Nail polish remover odor
        - Musty, wet cardboard smell (cork taint)
        - Overly harsh or bitter taste
        """
        documents.append(Document(
            page_content=tasting_guide,
            metadata={"source": "tasting_guide", "topic": "wine_tasting"}
        ))
        
        # Document 11: Food Pairing Guide
        food_pairing = """
        # Red Wine Food Pairing Guide
        
        ## General Pairing Principles
        
        ### Match Intensity
        - Light wines → Light dishes
        - Bold wines → Rich, hearty dishes
        
        ### Consider Components
        - **Tannins**: Cut through fatty foods
        - **Acidity**: Complements tomato-based dishes
        - **Alcohol**: Pairs with richer preparations
        
        ## Pairing by Wine Quality Level
        
        ### Quality 3-4 (Everyday Wines)
        Best uses:
        - Cooking wines for sauces
        - Casual BBQ and grilling
        - Simple pasta dishes
        - Pizza nights
        
        ### Quality 5-6 (Good Table Wines)
        Perfect pairings:
        - Grilled steaks and burgers
        - Roasted chicken
        - Beef stew and braises
        - Hard aged cheeses
        - Mushroom dishes
        
        ### Quality 7-8 (Premium Wines)
        Ideal matches:
        - Prime cuts of beef
        - Lamb dishes
        - Duck and game birds
        - Truffle preparations
        - Aged Parmesan and Pecorino
        
        ## Classic Red Wine Pairings
        
        ### High Alcohol Wines (12%+)
        - Rich red meats
        - Aged cheeses
        - Chocolate desserts
        - Roasted vegetables
        
        ### Moderate Alcohol Wines (10-12%)
        - Pasta with meat sauce
        - Grilled vegetables
        - Light game
        - Semi-hard cheeses
        
        ### Lower Alcohol Wines (<10%)
        - Light appetizers
        - Charcuterie
        - Mild cheeses
        - Vegetarian dishes
        
        ## Foods to Avoid with Red Wine
        - Very spicy dishes (overwhelm the wine)
        - Delicate fish (tannins clash)
        - Very sweet desserts (unless wine is sweeter)
        - Strong vinegar-based dressings
        """
        documents.append(Document(
            page_content=food_pairing,
            metadata={"source": "food_pairing", "topic": "food_pairing"}
        ))
        
        # Document 12: Wine Storage and Serving
        storage_serving = """
        # Red Wine Storage and Serving Guide
        
        ## Proper Storage Conditions
        
        ### Temperature
        - Ideal: 55°F (13°C)
        - Acceptable range: 45-65°F (7-18°C)
        - Avoid temperature fluctuations
        
        ### Humidity
        - Ideal: 60-70%
        - Prevents cork drying
        - Maintains seal integrity
        
        ### Light
        - Store in darkness
        - UV light damages wine
        - Avoid fluorescent lighting
        
        ### Position
        - Store bottles horizontally
        - Keeps cork moist
        - Prevents oxidation
        
        ## Serving Temperatures
        
        ### Light-bodied Red Wines
        - Temperature: 55-60°F (13-16°C)
        - Slight chill enhances freshness
        - Serve after 30 min in fridge
        
        ### Medium-bodied Red Wines  
        - Temperature: 60-65°F (16-18°C)
        - Room temperature or slightly below
        - Most red wines fall here
        
        ### Full-bodied Red Wines
        - Temperature: 63-68°F (17-20°C)
        - Just below room temperature
        - Allows complex aromas to emerge
        
        ## Decanting Guidelines
        
        ### When to Decant
        - Young, tannic wines: 1-2 hours
        - Mature wines: 30 minutes
        - Very old wines: Just before serving
        
        ### Benefits of Decanting
        - Separates sediment
        - Aerates the wine
        - Softens tannins
        - Opens up aromas
        
        ## Glassware
        
        ### Ideal Red Wine Glass Features
        - Large bowl for aeration
        - Tapered rim to concentrate aromas
        - Thin rim for pleasant drinking
        - Clear glass to observe color
        
        ### Fill Level
        - Fill to widest point of bowl
        - Usually 1/3 to 1/2 full
        - Allows swirling without spilling
        """
        documents.append(Document(
            page_content=storage_serving,
            metadata={"source": "storage_serving", "topic": "wine_care"}
        ))
        
        # Document 13: FAQ
        faq = """
        # Frequently Asked Questions About Red Wine
        
        ## General Questions
        
        **Q: What makes a wine "good quality"?**
        A: Quality wine exhibits balance between acidity, tannins, alcohol, and fruit.
        Key indicators include higher alcohol content (11-14%), moderate sulphates,
        low volatile acidity, and pleasant citric acid levels. The best wines
        show complexity, harmony, and a long finish.
        
        **Q: How long can I store red wine?**
        A: It depends on quality. Lower quality wines (3-5) should be consumed
        within 1-2 years. Quality 6-7 wines can age 3-5 years. Premium quality
        8 wines may improve over 5-10+ years with proper storage.
        
        **Q: What does "dry" wine mean?**
        A: A dry wine has little to no residual sugar (less than 4 g/L).
        Most red wines in our database are dry. The term describes the
        absence of sweetness, not the texture.
        
        **Q: Why does red wine make my mouth feel dry?**
        A: That's caused by tannins, which are natural compounds from grape
        skins, seeds, and sometimes oak barrels. Tannins bind to proteins
        in saliva, creating that astringent, drying sensation.
        
        ## About Wine Components
        
        **Q: Is higher alcohol always better?**
        A: Not necessarily, but our data shows a positive correlation between
        alcohol (11-14%) and quality. However, balance is key - alcohol should
        integrate with other elements, not dominate.
        
        **Q: What causes that vinegar smell in wine?**
        A: High volatile acidity, primarily acetic acid. This indicates
        bacterial contamination or oxidation. Our analysis shows wines with
        lower volatile acidity consistently rate higher in quality.
        
        **Q: Are sulfites bad for you?**
        A: Sulfites (sulfur dioxide) are natural preservatives in wine.
        Only about 1% of people have true sulfite sensitivity. The amounts
        in wine are generally safe for most people.
        
        **Q: What does pH tell us about wine?**
        A: pH measures acidity on a scale of 0-14. Wine typically ranges
        3.0-4.0. Lower pH (higher acidity) wines taste more crisp and age
        better. Our dataset averages around pH 3.3.
        
        ## Tasting and Serving
        
        **Q: Should I let red wine "breathe"?**
        A: Yes, especially young wines with strong tannins. Exposure to air
        softens tannins and allows aromas to develop. Decant for 30 minutes
        to 2 hours depending on the wine's age and style.
        
        **Q: What temperature should I serve red wine?**
        A: Slightly below room temperature, around 60-68°F (16-20°C).
        Light reds can be served slightly cooler. Never serve red wine
        too warm as it emphasizes alcohol harshness.
        
        **Q: How do I know if a wine has gone bad?**
        A: Look for brownish color, vinegar or musty smells, flat taste,
        or fizzing in still wines. Trust your senses - if it smells or
        tastes unpleasant, don't drink it.
        """
        documents.append(Document(
            page_content=faq,
            metadata={"source": "faq", "topic": "frequently_asked_questions"}
        ))
        
        return documents
    
    def get_documents(self) -> List[Document]:
        """Return all documents for the knowledge base."""
        return self.documents


class WineSommelierChatbot:
    """
    Main chatbot class that integrates all components.
    """
    
    def __init__(
        self,
        csv_path: str,
        anthropic_api_key: Optional[str] = None,
        embedding_model: str = "all-MiniLM-L6-v2"
    ):
        """
        Initialize the Wine Sommelier Chatbot.
        
        Args:
            csv_path: Path to the wine quality CSV file
            anthropic_api_key: Anthropic API key (or set ANTHROPIC_API_KEY env var)
            embedding_model: HuggingFace embedding model name
        """
        # Set API key
        if anthropic_api_key:
            os.environ["ANTHROPIC_API_KEY"] = anthropic_api_key
        
        # Initialize components
        print("Initializing Wine Sommelier Chatbot...")
        
        print("  - Loading wine dataset...")
        self.analyzer = WineDataAnalyzer(csv_path)
        
        print("  -Creating knowledge base...")
        self.knowledge_base = WineKnowledgeBase(self.analyzer)
        
        print("  -Setting up embeddings...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        print("  -Building vector store...")
        self.vectorstore = self._create_vectorstore()
        
        print("  -Initializing LLM...")
        self.llm = ChatAnthropic(
            model="claude-sonnet-4-20250514",
            temperature=0.7,
            max_tokens=1024
        )
        
        print("  -Setting up conversation chain...")
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer"
        )
        
        self.chain = self._create_chain()
        
        print("Wine Sommelier Chatbot ready!")
        
    def _create_vectorstore(self) -> FAISS:
        """Create FAISS vector store from knowledge base documents."""
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", ". ", " ", ""]
        )

        documents = self.knowledge_base.get_documents()
        splits = text_splitter.split_documents(documents)

        vectorstore = FAISS.from_documents(
            documents=splits,
            embedding=self.embeddings
        )

        return vectorstore
    
    def _create_chain(self) -> ConversationalRetrievalChain:
        """Create the conversational retrieval chain."""
        
        # Custom prompt for the sommelier persona
        system_template = """You are an expert wine sommelier with deep knowledge of red wines.
        You have access to detailed information about wine chemistry, quality factors, 
        tasting notes, food pairings, and wine care.
        
        Use the following context to answer the user's question. If the context doesn't 
        contain relevant information, use your general wine knowledge but mention that 
        you're drawing from general expertise.
        
        Always be helpful, informative, and enthusiastic about wine. Provide specific 
        recommendations when appropriate, and explain wine concepts in an accessible way.
        
        If asked about specific wine recommendations, reference the quality profiles and 
        characteristics from the dataset.
        
        Context: {context}
        
        Chat History: {chat_history}
        
        Question: {question}
        
        Helpful Answer:"""
        
        prompt = PromptTemplate(
            template=system_template,
            input_variables=["context", "chat_history", "question"]
        )
        
        chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=self.vectorstore.as_retriever(
                search_type="mmr",  # Maximum Marginal Relevance
                search_kwargs={"k": 4, "fetch_k": 8}
            ),
            memory=self.memory,
            combine_docs_chain_kwargs={"prompt": prompt},
            return_source_documents=True,
            verbose=False
        )
        
        return chain
    
    def chat(self, user_input: str) -> str:
        """
        Process user input and return the chatbot response.
        
        Args:
            user_input: The user's question or message
            
        Returns:
            The chatbot's response
        """
        try:
            result = self.chain.invoke({"question": user_input})
            return result["answer"]
        except Exception as e:
            return f"I apologize, but I encountered an error: {str(e)}. Please try rephrasing your question."
    
    def get_wine_stats(self) -> str:
        """Return formatted wine statistics."""
        stats = self.analyzer.get_statistics()
        return json.dumps(stats, indent=2)
    
    def get_quality_recommendation(self, min_quality: int = 6) -> str:
        """Get wine recommendations above a certain quality threshold."""
        recommendations = self.analyzer.get_wine_recommendations(
            {"min_quality": min_quality}
        )
        return f"Found {len(recommendations)} wines with quality {min_quality} or higher."
    
    def reset_conversation(self):
        """Clear the conversation history."""
        self.memory.clear()
        print("Conversation history cleared.")


def main():
    """Main function to run the chatbot interactively."""
    import sys
    
    # Check for API key
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("\n  Warning: ANTHROPIC_API_KEY not set!")
        print("Please set your API key:")
        print("  export ANTHROPIC_API_KEY='your-api-key'")
        print("\nOr pass it when initializing the chatbot.\n")
        sys.exit(1)
    
    # Initialize chatbot
    csv_path = "winequality-red.csv"  # Update path as needed
    
    try:
        chatbot = WineSommelierChatbot(csv_path)
    except FileNotFoundError:
        print(f"\n Error: Could not find '{csv_path}'")
        print("Please ensure the wine quality CSV file is in the current directory.")
        sys.exit(1)
    
    # Print welcome message
    print("\n" + "="*60)
    print(" Welcome to the Wine Sommelier Chatbot! ")
    print("="*60)
    print("\nI'm your personal wine expert. Ask me anything about:")
    print("  - Red wine quality and characteristics")
    print("  - Wine chemistry and what affects taste")
    print("  - Food pairing suggestions")
    print("  - Wine tasting and serving tips")
    print("  - Storage recommendations")
    print("\nCommands:")
    print("  'stats'  - Show wine dataset statistics")
    print("  'reset'  - Clear conversation history")
    print("  'quit'   - Exit the chatbot")
    print("\n" + "-"*60 + "\n")
    
    # Main conversation loop
    while True:
        try:
            user_input = input("You: ").strip()
            
            if not user_input:
                continue
            
            # Handle special commands
            if user_input.lower() == 'quit':
                print("\n Thank you for using Wine Sommelier! Cheers! \n")
                break
            elif user_input.lower() == 'stats':
                print(f"\nSommelier: Here are the wine statistics:\n{chatbot.get_wine_stats()}\n")
                continue
            elif user_input.lower() == 'reset':
                chatbot.reset_conversation()
                print("\nSommelier: Conversation reset. How can I help you?\n")
                continue
            
            # Get chatbot response
            response = chatbot.chat(user_input)
            print(f"\nSommelier: {response}\n")
            
        except KeyboardInterrupt:
            print("\n\n Goodbye! Enjoy your wine! \n")
            break
        except Exception as e:
            print(f"\nError: {str(e)}\n")


if __name__ == "__main__":
    main()