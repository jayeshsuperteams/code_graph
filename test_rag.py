from rag import rag_query

def test_rag_system():
    # Basic functionality test
    test_question = "What is the main topic of this document?"
    
    try:
        response = rag_query(test_question)
        print("âœ… System working!")
        print(f"Question: {test_question}")
        print(f"Response: {response}")
    except Exception as e:
        print(f"âŒ Error: {str(e)}")

if __name__ == "__main__":
    test_rag_system()