'use client'

import { useState, useEffect } from 'react'
import Link from 'next/link'
import Dashboard from './components/Dashboard'
import CampaignForm from './components/CampaignForm'
import PlatformConnections from './components/PlatformConnections'
import Analytics from './components/Analytics'

export default function Home() {
  const [activeTab, setActiveTab] = useState('dashboard')

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <header className="bg-white shadow-sm">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4">
          <div className="flex justify-between items-center">
            <div>
              <h1 className="text-2xl font-bold text-gray-900">
                🚀 Agentic AI Social Media Automation
              </h1>
              <p className="text-sm text-gray-600">
                Multi-platform automation powered by Gemini AI
              </p>
            </div>
            <nav className="flex space-x-4">
              <button
                onClick={() => setActiveTab('dashboard')}
                className={`px-4 py-2 rounded-lg transition-colors ${
                  activeTab === 'dashboard'
                    ? 'bg-primary-600 text-white'
                    : 'text-gray-700 hover:bg-gray-100'
                }`}
              >
                Dashboard
              </button>
              <button
                onClick={() => setActiveTab('campaigns')}
                className={`px-4 py-2 rounded-lg transition-colors ${
                  activeTab === 'campaigns'
                    ? 'bg-primary-600 text-white'
                    : 'text-gray-700 hover:bg-gray-100'
                }`}
              >
                Campaigns
              </button>
              <button
                onClick={() => setActiveTab('platforms')}
                className={`px-4 py-2 rounded-lg transition-colors ${
                  activeTab === 'platforms'
                    ? 'bg-primary-600 text-white'
                    : 'text-gray-700 hover:bg-gray-100'
                }`}
              >
                Platforms
              </button>
              <button
                onClick={() => setActiveTab('analytics')}
                className={`px-4 py-2 rounded-lg transition-colors ${
                  activeTab === 'analytics'
                    ? 'bg-primary-600 text-white'
                    : 'text-gray-700 hover:bg-gray-100'
                }`}
              >
                Analytics
              </button>
            </nav>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {activeTab === 'dashboard' && <Dashboard />}
        {activeTab === 'campaigns' && <CampaignForm />}
        {activeTab === 'platforms' && <PlatformConnections />}
        {activeTab === 'analytics' && <Analytics />}
      </main>
    </div>
  )
}

