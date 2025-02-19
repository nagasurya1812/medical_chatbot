import React, { useState, useRef, useEffect } from "react";
import axios from "axios";
import confetti from "canvas-confetti"; 
import "./App.css"; 
import botAvatar from "./images/bot (2).jpeg"; 
import userAvatar from "./images/human.jpeg"; 

function App() {
  const [message, setMessage] = useState("");
  const [messages, setMessages] = useState([]);
  const [loading, setLoading] = useState(false);
  const chatEndRef = useRef(null);

  useEffect(() => {
    confetti({
      particleCount: 100,
      spread: 80,
      origin: { y: 0.6},
    });
  }, []);

  const sendMessage = async () => {
    if (!message.trim()) return;

    const userMessage = { text: message, sender: "user" };
    setMessages((prev) => [...prev, userMessage]);
    setLoading(true);
    setMessage("");

    try {
      const res = await axios.post("http://localhost:5000/chat", { message });
      const botMessage = { text: res.data.response, sender: "bot" };
      setMessages((prev) => [...prev, botMessage]);
    } catch (error) {
      setMessages((prev) => [
        ...prev,
        { text: "Error: Server not responding", sender: "bot" },
      ]);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  return (
    <div className="app-container">
      {/* Header */}
      <header className="header">Gale Encyclopedia of Medicine</header>

      {/* Main Chat Section */}
      <div className="chat-container">
        <h1 className="chat-title">Medical Chat Bot</h1>
        
     <p className="chat-subtitle">
          Clarify your medical-related doubts here! 🤔💡🩺
      </p>


        <div className="chat-box">
          {messages.map((msg, index) => (
            <div key={index} className={`message-container ${msg.sender}`}>
              <img
                src={msg.sender === "bot" ? botAvatar : userAvatar}
                alt="avatar"
                className="avatar"
              />
              <div className="message">{msg.text}</div>
            </div>
          ))}
          {loading && (
            <div className="message-container bot">
              <img src={botAvatar} alt="avatar" className="avatar" />
              <div className="message">...</div>
            </div>
          )}
          <div ref={chatEndRef} />
        </div>

        <div className="input-area">
          <input
            type="text"
            placeholder="Type a message..."
            value={message}
            onChange={(e) => setMessage(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && sendMessage()}
          />
          <button onClick={sendMessage} disabled={loading}>
            &#9658;
          </button>
        </div>
      </div>

      {/* Footer */}
      <footer className="footer">&copy; Reserved by NAGASURYA 2025-2026</footer>
    </div>
  );
}

export default App;
