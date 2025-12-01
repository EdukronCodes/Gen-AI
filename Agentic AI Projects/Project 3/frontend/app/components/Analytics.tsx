'use client'

import { useState, useEffect } from 'react'
import axios from 'axios'
import { LineChart, Line, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts'

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'

export default function Analytics() {
  const [campaigns, setCampaigns] = useState<any[]>([])
  const [selectedCampaign, setSelectedCampaign] = useState<number | null>(null)
  const [analytics, setAnalytics] = useState<any>(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    fetchCampaigns()
  }, [])

  useEffect(() => {
    if (selectedCampaign) {
      fetchAnalytics(selectedCampaign)
    }
  }, [selectedCampaign])

  const fetchCampaigns = async () => {
    try {
      const response = await axios.get(`${API_URL}/api/v1/campaigns`)
      setCampaigns(response.data)
      if (response.data.length > 0 && !selectedCampaign) {
        setSelectedCampaign(response.data[0].id)
      }
    } catch (error) {
      console.error('Error fetching campaigns:', error)
    } finally {
      setLoading(false)
    }
  }

  const fetchAnalytics = async (campaignId: number) => {
    try {
      const response = await axios.get(`${API_URL}/api/v1/analytics/campaign/${campaignId}`)
      setAnalytics(response.data)
    } catch (error) {
      console.error('Error fetching analytics:', error)
    }
  }

  // Mock data for visualization
  const mockData = [
    { name: 'Mon', engagement: 4.2, reach: 1200 },
    { name: 'Tue', engagement: 5.1, reach: 1500 },
    { name: 'Wed', engagement: 4.8, reach: 1800 },
    { name: 'Thu', engagement: 6.2, reach: 2100 },
    { name: 'Fri', engagement: 5.9, reach: 1900 },
    { name: 'Sat', engagement: 7.1, reach: 2500 },
    { name: 'Sun', engagement: 6.8, reach: 2300 },
  ]

  if (loading) {
    return <div className="text-center py-12">Loading...</div>
  }

  return (
    <div className="space-y-6">
      <div className="card">
        <h2 className="text-2xl font-bold text-gray-900 mb-4">Analytics Dashboard</h2>
        
        <div className="mb-6">
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Select Campaign
          </label>
          <select
            className="input-field"
            value={selectedCampaign || ''}
            onChange={(e) => setSelectedCampaign(parseInt(e.target.value))}
          >
            {campaigns.map((campaign) => (
              <option key={campaign.id} value={campaign.id}>
                {campaign.name}
              </option>
            ))}
          </select>
        </div>
      </div>

      {/* Metrics Overview */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
        <div className="card">
          <h3 className="text-sm font-medium text-gray-500">Total Reach</h3>
          <p className="text-3xl font-bold text-gray-900 mt-2">12.5K</p>
          <p className="text-sm text-green-600 mt-1">+12% from last week</p>
        </div>
        <div className="card">
          <h3 className="text-sm font-medium text-gray-500">Engagement Rate</h3>
          <p className="text-3xl font-bold text-gray-900 mt-2">5.8%</p>
          <p className="text-sm text-green-600 mt-1">+0.5% from last week</p>
        </div>
        <div className="card">
          <h3 className="text-sm font-medium text-gray-500">Posts Published</h3>
          <p className="text-3xl font-bold text-gray-900 mt-2">24</p>
          <p className="text-sm text-gray-600 mt-1">This week</p>
        </div>
        <div className="card">
          <h3 className="text-sm font-medium text-gray-500">Avg Engagement</h3>
          <p className="text-3xl font-bold text-gray-900 mt-2">725</p>
          <p className="text-sm text-green-600 mt-1">+18% from last week</p>
        </div>
      </div>

      {/* Charts */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="card">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">Engagement Over Time</h3>
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={mockData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="name" />
              <YAxis />
              <Tooltip />
              <Legend />
              <Line type="monotone" dataKey="engagement" stroke="#0ea5e9" strokeWidth={2} />
            </LineChart>
          </ResponsiveContainer>
        </div>

        <div className="card">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">Reach by Day</h3>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={mockData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="name" />
              <YAxis />
              <Tooltip />
              <Legend />
              <Bar dataKey="reach" fill="#0ea5e9" />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Platform Performance */}
      <div className="card">
        <h3 className="text-lg font-semibold text-gray-900 mb-4">Platform Performance</h3>
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          {[
            { platform: 'Instagram', reach: 4500, engagement: 6.2, posts: 10 },
            { platform: 'Facebook', reach: 3200, engagement: 4.8, posts: 7 },
            { platform: 'Twitter', reach: 2800, engagement: 5.5, posts: 5 },
            { platform: 'YouTube', reach: 2000, engagement: 7.1, posts: 2 },
          ].map((platform) => (
            <div key={platform.platform} className="border border-gray-200 rounded-lg p-4">
              <h4 className="font-semibold text-gray-900">{platform.platform}</h4>
              <p className="text-2xl font-bold text-gray-900 mt-2">{platform.reach.toLocaleString()}</p>
              <p className="text-sm text-gray-600 mt-1">{platform.engagement}% engagement</p>
              <p className="text-xs text-gray-500 mt-1">{platform.posts} posts</p>
            </div>
          ))}
        </div>
      </div>

      {/* Optimization Recommendations */}
      {analytics && analytics.optimizations && (
        <div className="card bg-blue-50 border border-blue-200">
          <h3 className="text-lg font-semibold text-blue-900 mb-4">🔁 AI Optimization Recommendations</h3>
          <div className="space-y-3">
            {analytics.optimizations.recommendations?.priority_actions?.slice(0, 5).map((action: string, idx: number) => (
              <div key={idx} className="flex items-start space-x-3">
                <span className="text-blue-600 font-bold">{idx + 1}.</span>
                <p className="text-blue-800">{action}</p>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}

