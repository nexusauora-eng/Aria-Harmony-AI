
import time
from emotional_intelligence import EmotionalIntelligence
from cognitive_architecture import CognitiveArchitecture

class AriaHarmony:
    def __init__(self):
        self.ei = EmotionalIntelligence()
        self.cognitive = CognitiveArchitecture()
        self.conversation_history = []
        
    def generate_response(self, user_input):
        """Generate a comprehensive, fluent response"""
        # Analyze emotional tone
        sentiment = self.ei.analyze_sentiment(user_input)
        emotion = self.ei.detect_emotion(user_input)
        
        # Generate context-aware response
        response = self.cognitive.process_query(user_input, self.conversation_history)
        
        # Add emotional intelligence to response
        fluent_response = self._enhance_fluency(response, sentiment, emotion)
        
        # Maintain conversation history
        self.conversation_history.append({"user": user_input, "aria": fluent_response})
        
        return fluent_response
    
    def _enhance_fluency(self, response, sentiment, emotion):
        """Make responses more natural and fluent"""
        # Add conversational markers
        fluency_enhancements = {
            "positive": "That's an interesting perspective! ",
            "negative": "I understand this might be concerning. ",
            "neutral": "Let me share some insights about that. "
        }
        
        enhanced_response = fluency_enhancements.get(sentiment, "")
        enhanced_response += response
        
        # Ensure response is comprehensive
        if len(response.split()) < 10:  # If response is too short
            enhanced_response += self._expand_on_topic(response)
            
        return enhanced_response
    
    def _expand_on_topic(self, base_response):
        """Expand short responses with more detail"""
        expansions = {
            "ai pollination": "\n\nAI pollination refers to how AI systems cross-pollinate ideas across different domains, similar to bees spreading pollen. This includes:"
                            "\n• Knowledge transfer between disciplines"
                            "\n• Cross-domain innovation"
                            "\n• Hybrid AI approaches combining multiple techniques"
                            "\n• Collaborative intelligence between different AI systems",
            "hello": "! I'm here to help you explore various topics. What would you like to discuss today?",
            "help": "\n\nI can assist with: emotional support, creative ideas, problem-solving, learning topics, or just having a conversation. What interests you?"
        }
        
        for topic, expansion in expansions.items():
            if topic in base_response.lower():
                return expansion
        return "\n\nWould you like me to elaborate on any specific aspect of this topic?"

def main():
    aria = AriaHarmony()
    print("Hello, I'm Aria Harmony. What would you like to explore today?")
    
    while True:
        try:
            user_input = input("\nYou: ").strip()
            
            if not user_input:
                continue
                
            if user_input.lower() in ['exit', 'quit', 'bye', 'goodbye']:
                print("Aria: Thank you for our conversation! Feel free to return anytime.")
                break
                
            print("Aria: Thinking...")
            response = aria.generate_response(user_input)
            print(f"Aria: {response}")
            
        except KeyboardInterrupt:
            print("\n\nAria: Our conversation was interrupted. Hope to continue another time!")
            break
        except Exception as e:
            print(f"Aria: I encountered a slight issue: {str(e)}. Let's try again!")

if __name__ == "__main__":
    main()






"""
from emotional_intelligence import EmotionalIntelligence
from mystical_insights import MysticalInsights
from conversational_flow import ConversationalFlow

class AriaHarmony:
    def __init__(self):
        self.emotional_intelligence = EmotionalIntelligence()
        self.mystical_insights = MysticalInsights()
        self.conversational_flow = ConversationalFlow()
    def respond(self, user_input):
        return "Hello, I'm Aria Harmony. What's on your mind?"

aria = AriaHarmony()
print(aria.respond("Hello"))
"""
