import React, { useState, useRef, useEffect } from 'react';
import axios from 'axios';
import './App.css'; // We will add simple styles below

function App() {
  const [messages, setMessages] = useState([
    { role: 'bot', text: 'Hello! I have analyzed the RFP documents. What would you like to know?' }
  ]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(scrollToBottom, [messages]);

  const handleSend = async () => {
    if (!input.trim()) return;

    const userMessage = { role: 'user', text: input };
    setMessages(prev => [...prev, userMessage]);
    setInput('');
    setLoading(true);

    try {
      const response = await axios.post('http://localhost:8000/chat', {
        question: userMessage.text
      });

      const botMessage = {
        role: 'bot',
        text: response.data.answer,
        sources: response.data.sources
      };

      setMessages(prev => [...prev, botMessage]);
    } catch (error) {
      setMessages(prev => [...prev, { role: 'bot', text: 'Sorry, something went wrong connecting to the server.' }]);
    } finally {
      setLoading(false);
    }
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter') handleSend();
  };

  return (
    <div className="app-container">
      <header className="chat-header">
        <h1>CSA RFP Analyzer ChatBot</h1>
      </header>

      <div className="chat-window">
        {messages.map((msg, index) => (
          <div key={index} className={`message ${msg.role}`}>
            <div className="message-bubble">
              <p>{msg.text}</p>
              {msg.sources && msg.sources.length > 0 && (
                <div className="sources">
                  <small>Source: {msg.sources.map(s => s.split('/').pop()).join(', ')}</small>
                </div>
              )}
            </div>
          </div>
        ))}
        {loading && <div className="message bot"><div className="message-bubble">Analyzing PDFs...</div></div>}
        <div ref={messagesEndRef} />
      </div>

      <div className="input-area">
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyPress={handleKeyPress}
          placeholder="Ask about Scope, SLAs, or Requirements..."
        />
        <button onClick={handleSend} disabled={loading}>Send</button>
      </div>
    </div>
  );
}

export default App;