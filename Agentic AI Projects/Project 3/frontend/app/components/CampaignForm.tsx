'use client'

import { useState } from 'react'
import axios from 'axios'

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'

export default function CampaignForm() {
  const [loading, setLoading] = useState(false)
  const [success, setSuccess] = useState(false)
  const [formData, setFormData] = useState({
    name: '',
    description: '',
    goal: '',
    target_platforms: [] as string[],
    duration_days: 7,
    content_themes: [] as string[],
    target_audience: {}
  })

  const platforms = ['instagram', 'facebook', 'twitter', 'youtube']
  const themes = ['Educational', 'Entertainment', 'Promotional', 'Inspirational', 'Behind-the-scenes']

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setLoading(true)
    setSuccess(false)

    try {
      const response = await axios.post(`${API_URL}/api/v1/campaigns`, {
        user_id: 1, // In production, get from auth
        ...formData
      })

      setSuccess(true)
      setFormData({
        name: '',
        description: '',
        goal: '',
        target_platforms: [],
        duration_days: 7,
        content_themes: [],
        target_audience: {}
      })

      setTimeout(() => setSuccess(false), 5000)
    } catch (error: any) {
      console.error('Error creating campaign:', error)
      alert('Error creating campaign: ' + (error.response?.data?.detail || error.message))
    } finally {
      setLoading(false)
    }
  }

  const togglePlatform = (platform: string) => {
    setFormData(prev => ({
      ...prev,
      target_platforms: prev.target_platforms.includes(platform)
        ? prev.target_platforms.filter(p => p !== platform)
        : [...prev.target_platforms, platform]
    }))
  }

  const toggleTheme = (theme: string) => {
    setFormData(prev => ({
      ...prev,
      content_themes: prev.content_themes.includes(theme)
        ? prev.content_themes.filter(t => t !== theme)
        : [...prev.content_themes, theme]
    }))
  }

  return (
    <div className="max-w-3xl mx-auto">
      <div className="card">
        <h2 className="text-2xl font-bold text-gray-900 mb-6">Create New Campaign</h2>
        
        {success && (
          <div className="mb-4 p-4 bg-green-100 border border-green-400 text-green-700 rounded-lg">
            Campaign created successfully! AI agents are now generating your content strategy.
          </div>
        )}

        <form onSubmit={handleSubmit} className="space-y-6">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Campaign Name
            </label>
            <input
              type="text"
              required
              className="input-field"
              value={formData.name}
              onChange={(e) => setFormData({ ...formData, name: e.target.value })}
              placeholder="e.g., Data Science Course Promotion"
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Description
            </label>
            <textarea
              className="input-field"
              rows={3}
              value={formData.description}
              onChange={(e) => setFormData({ ...formData, description: e.target.value })}
              placeholder="Brief description of your campaign"
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Goal / Objective
            </label>
            <textarea
              required
              className="input-field"
              rows={4}
              value={formData.goal}
              onChange={(e) => setFormData({ ...formData, goal: e.target.value })}
              placeholder="e.g., Promote my Data Science course for the next 7 days. Focus on engagement and lead generation."
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Target Platforms
            </label>
            <div className="flex flex-wrap gap-3">
              {platforms.map((platform) => (
                <button
                  key={platform}
                  type="button"
                  onClick={() => togglePlatform(platform)}
                  className={`px-4 py-2 rounded-lg border transition-colors ${
                    formData.target_platforms.includes(platform)
                      ? 'bg-primary-600 text-white border-primary-600'
                      : 'bg-white text-gray-700 border-gray-300 hover:bg-gray-50'
                  }`}
                >
                  {platform.charAt(0).toUpperCase() + platform.slice(1)}
                </button>
              ))}
            </div>
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Duration (days)
            </label>
            <input
              type="number"
              min="1"
              max="30"
              className="input-field"
              value={formData.duration_days}
              onChange={(e) => setFormData({ ...formData, duration_days: parseInt(e.target.value) })}
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Content Themes
            </label>
            <div className="flex flex-wrap gap-3">
              {themes.map((theme) => (
                <button
                  key={theme}
                  type="button"
                  onClick={() => toggleTheme(theme)}
                  className={`px-4 py-2 rounded-lg border transition-colors ${
                    formData.content_themes.includes(theme)
                      ? 'bg-primary-600 text-white border-primary-600'
                      : 'bg-white text-gray-700 border-gray-300 hover:bg-gray-50'
                  }`}
                >
                  {theme}
                </button>
              ))}
            </div>
          </div>

          <button
            type="submit"
            disabled={loading}
            className="w-full btn-primary py-3 text-lg font-semibold disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {loading ? 'Creating Campaign...' : 'Create Campaign & Generate Strategy'}
          </button>
        </form>

        <div className="mt-6 p-4 bg-blue-50 rounded-lg">
          <p className="text-sm text-blue-800">
            <strong>🤖 AI Agents at Work:</strong> Once created, our AI agents will automatically:
            <ul className="list-disc list-inside mt-2 space-y-1">
              <li>Create a comprehensive content strategy</li>
              <li>Generate captions, scripts, and content</li>
              <li>Design visual content ideas</li>
              <li>Optimize posting schedules</li>
              <li>Prepare posts for publishing</li>
            </ul>
          </p>
        </div>
      </div>
    </div>
  )
}

