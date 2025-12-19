import React, { useState, useEffect } from 'react';
import BrowserOnly from '@docusaurus/BrowserOnly';

const RAGChatbot = () => {
  const [isOpen, setIsOpen] = useState(false);
  const [messages, setMessages] = useState([
    { id: 1, text: 'Hello! I\'m your Physical AI & Humanoid Robotics assistant. Ask me anything about the textbook content.', sender: 'bot' }
  ]);
  const [inputValue, setInputValue] = useState('');
  const [isLoading, setIsLoading] = useState(false);

  // Function to send message to RAG backend
  const sendMessage = async (message) => {
    setIsLoading(true);

    try {
      // Use the local RAG API endpoint for development first
      // In a real deployment, this would be your actual API endpoint
      const ragApiUrl = typeof window !== 'undefined'
        ? 'http://localhost:8000/query/' // Local development API
        : 'https://physical-ai-humanoid-robotics-rag.onrender.com/query/'; // Production API

      // Format request to match the API schema (using question instead of query)
      const response = await fetch(ragApiUrl, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          question: message,
          max_results: 5,
          grounding_threshold: 0.7
        }),
      });

      if (response.ok) {
        const data = await response.json();
        // Handle both possible response formats (answer or response field)
        // Only return the fallback if the API specifically returns that message
        const apiResponse = data.answer || data.response;
        if (apiResponse && apiResponse !== 'I found some information about that topic in the textbook.') {
          return apiResponse;
        } else {
          // If the API returns the generic fallback, try to provide more specific help
          return 'I found some information about that topic in the textbook. For detailed answers, please check the relevant module sections (Module 1: ROS 2, Module 2: Digital Twin, Module 3: AI-Robot Brain, Module 4: VLA).';
        }
      } else {
        console.error('RAG API error:', response.status, response.statusText);
        // Fallback response if API is not available
        return 'API server error. Please make sure the RAG API server is running with "python run_server.py" in the rag-system directory.';
      }
    } catch (error) {
      console.error('Error calling RAG API:', error);
      // Check if it's a network error (server not running)
      if (error.name === 'TypeError' && error.message.includes('fetch')) {
        return 'API server is not running. Please start the RAG API server with "python run_server.py" in the rag-system directory.';
      }
      // Fallback response in case of network error
      return 'Network error. Please make sure the RAG API server is running.';
    } finally {
      setIsLoading(false);
    }
  };

  const handleSendMessage = async () => {
    if (!inputValue.trim()) return;

    const userMessage = { id: Date.now(), text: inputValue, sender: 'user' };
    setMessages(prev => [...prev, userMessage]);
    setInputValue('');

    const botResponse = await sendMessage(inputValue);
    const botMessage = { id: Date.now() + 1, text: botResponse, sender: 'bot' };
    setMessages(prev => [...prev, botMessage]);
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  return (
    <BrowserOnly>
      {() => (
        <div className={`rag-chatbot ${isOpen ? 'open' : 'closed'}`}>
          <button
            className="chatbot-toggle"
            onClick={() => setIsOpen(!isOpen)}
          >
            {isOpen ? '✕' : '🤖'}
          </button>

          {isOpen && (
            <div className="chatbot-container">
              <div className="chatbot-header">
                <h4>Robotics Assistant</h4>
              </div>

              <div className="chatbot-messages">
                {messages.map((message) => (
                  <div
                    key={message.id}
                    className={`message ${message.sender}`}
                  >
                    <div className="message-text">{message.text}</div>
                  </div>
                ))}

                {isLoading && (
                  <div className="message bot">
                    <div className="message-text">...</div>
                  </div>
                )}
              </div>

              <div className="chatbot-input">
                <textarea
                  value={inputValue}
                  onChange={(e) => setInputValue(e.target.value)}
                  onKeyPress={handleKeyPress}
                  placeholder="Ask about robotics concepts..."
                  rows="2"
                />
                <button
                  onClick={handleSendMessage}
                  disabled={isLoading || !inputValue.trim()}
                  className="send-button"
                >
                  Send
                </button>
              </div>
            </div>
          )}

          <style jsx>{`
            .rag-chatbot {
              position: fixed;
              bottom: 20px;
              right: 20px;
              z-index: 1000;
            }

            .chatbot-toggle {
              width: 60px;
              height: 60px;
              border-radius: 50%;
              background: #2563eb;
              color: white;
              border: none;
              font-size: 24px;
              cursor: pointer;
              display: flex;
              align-items: center;
              justify-content: center;
              box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
              transition: all 0.3s ease;
            }

            .chatbot-toggle:hover {
              background: #1d4ed8;
              transform: scale(1.05);
            }

            .chatbot-container {
              width: 350px;
              height: 500px;
              background: white;
              border-radius: 12px;
              box-shadow: 0 8px 30px rgba(0, 0, 0, 0.2);
              display: flex;
              flex-direction: column;
              overflow: hidden;
              margin-bottom: 15px;
              border: 1px solid #e5e7eb;
            }

            .chatbot-header {
              background: #2563eb;
              color: white;
              padding: 15px;
              text-align: center;
            }

            .chatbot-header h4 {
              margin: 0;
              font-size: 16px;
            }

            .chatbot-messages {
              flex: 1;
              overflow-y: auto;
              padding: 15px;
              background: #f9fafb;
            }

            .message {
              margin-bottom: 12px;
              max-width: 80%;
            }

            .message.user {
              margin-left: auto;
              text-align: right;
            }

            .message.bot {
              margin-right: auto;
            }

            .message-text {
              display: inline-block;
              padding: 8px 12px;
              border-radius: 18px;
              font-size: 14px;
              line-height: 1.4;
            }

            .message.user .message-text {
              background: #2563eb;
              color: white;
            }

            .message.bot .message-text {
              background: #e5e7eb;
              color: #1f2937;
            }

            .chatbot-input {
              padding: 15px;
              border-top: 1px solid #e5e7eb;
              background: white;
              display: flex;
              gap: 8px;
            }

            .chatbot-input textarea {
              flex: 1;
              border: 1px solid #d1d5db;
              border-radius: 8px;
              padding: 8px 12px;
              resize: none;
              font-size: 14px;
              outline: none;
            }

            .chatbot-input textarea:focus {
              border-color: #2563eb;
              box-shadow: 0 0 0 2px rgba(37, 99, 235, 0.1);
            }

            .send-button {
              background: #2563eb;
              color: white;
              border: none;
              border-radius: 8px;
              padding: 8px 16px;
              cursor: pointer;
              font-size: 14px;
            }

            .send-button:hover:not(:disabled) {
              background: #1d4ed8;
            }

            .send-button:disabled {
              background: #9ca3af;
              cursor: not-allowed;
            }
          `}</style>
        </div>
      )}
    </BrowserOnly>
  );
};

export default RAGChatbot;