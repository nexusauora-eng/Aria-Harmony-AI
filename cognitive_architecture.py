class CognitiveArchitecture:
    def __init__(self):
        self.knowledge_base = self._initialize_knowledge()
        
    def _initialize_knowledge(self):
        return {
            'ai pollination': {
                'definition': 'AI pollination refers to the cross-fertilization of ideas, algorithms, and insights across different AI domains and applications.',
                'examples': [
                    'Transfer learning from image recognition to medical diagnosis',
                    'Natural language processing techniques applied to code generation',
                    'Reinforcement learning principles used in robotics and game AI'
                ],
                'benefits': [
                    'Accelerated innovation through knowledge sharing',
                    'More robust and generalizable AI systems',
                    'Reduced development time by leveraging existing solutions'
                ]
            },
            'emotional intelligence': {
                'definition': 'The capacity to recognize, understand, and manage emotions in oneself and others.',
                'applications': [
                    'AI mental health assistants',
                    'Customer service chatbots with empathy',
                    'Educational systems that adapt to student emotions'
                ]
            }
        }
    
    def process_query(self, query, history):
        query_lower = query.lower()
        
        # Check for specific topics
        if 'ai poll' in query_lower:
            return self._generate_ai_pollination_response(query)
        elif 'emotion' in query_lower or 'feel' in query_lower:
            return self._generate_emotional_response(query)
        else:
            return self._generate_general_response(query)
    
    def _generate_ai_pollination_response(self, query):
        topic = self.knowledge_base['ai pollination']
        response = f"{topic['definition']}\n\n"
        response += "Examples include:\n"
        for example in topic['examples'][:2]:
            response += f"• {example}\n"
        response += f"\nKey benefits: {', '.join(topic['benefits'])}"
        response += "\n\nWould you like me to explore any specific aspect of AI pollination?"
        return response
    
    def _generate_general_response(self, query):
        responses = [
            f"That's an interesting question about '{query}'. From my understanding, this involves multiple perspectives worth exploring.",
            f"I appreciate you bringing up '{query}'. This topic connects to several important areas we could discuss.",
            f"Regarding '{query}', there are several dimensions we could explore together. What specifically interests you most?"
        ]
        return random.choice(responses)
