import React from 'react'
import { Link } from 'react-router-dom'
import { Bot, Home, MessageSquare, Ticket } from 'lucide-react'

function Navbar() {
  return (
    <nav className="bg-blue-600 text-white shadow-lg">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex justify-between h-16">
          <div className="flex items-center">
            <Bot className="h-8 w-8 mr-2" />
            <span className="text-xl font-bold">Agentic AI Help Desk</span>
          </div>
          <div className="flex space-x-4 items-center">
            <Link to="/" className="flex items-center px-3 py-2 rounded hover:bg-blue-700">
              <Home className="h-5 w-5 mr-1" />
              Dashboard
            </Link>
            <Link to="/chatbot" className="flex items-center px-3 py-2 rounded hover:bg-blue-700">
              <MessageSquare className="h-5 w-5 mr-1" />
              Chatbot
            </Link>
            <Link to="/tickets" className="flex items-center px-3 py-2 rounded hover:bg-blue-700">
              <Ticket className="h-5 w-5 mr-1" />
              Tickets
            </Link>
          </div>
        </div>
      </div>
    </nav>
  )
}

export default Navbar


