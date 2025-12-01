'use client'

import { useState, useEffect } from 'react'
import axios from 'axios'

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'

export default function Dashboard() {
  const [campaigns, setCampaigns] = useState([])
  const [stats, setStats] = useState({
    totalCampaigns: 0,
    activeCampaigns: 0,
    totalPosts: 0,
    engagement: 0
  })
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    fetchDashboardData()
  }, [])

  const fetchDashboardData = async () => {
    try {
      const [campaignsRes] = await Promise.all([
        axios.get(`${API_URL}/api/v1/campaigns`)
      ])
      
      setCampaigns(campaignsRes.data.slice(0, 5))
      setStats({
        totalCampaigns: campaignsRes.data.length,
        activeCampaigns: campaignsRes.data.filter((c: any) => c.status === 'active').length,
        totalPosts: 0,
        engagement: 0
      })
    } catch (error) {
      console.error('Error fetching dashboard data:', error)
    } finally {
      setLoading(false)
    }
  }

  if (loading) {
    return <div className="text-center py-12">Loading...</div>
  }

  return (
    <div className="space-y-6">
      {/* Stats Grid */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
        <div className="card">
          <h3 className="text-sm font-medium text-gray-500">Total Campaigns</h3>
          <p className="text-3xl font-bold text-gray-900 mt-2">{stats.totalCampaigns}</p>
        </div>
        <div className="card">
          <h3 className="text-sm font-medium text-gray-500">Active Campaigns</h3>
          <p className="text-3xl font-bold text-primary-600 mt-2">{stats.activeCampaigns}</p>
        </div>
        <div className="card">
          <h3 className="text-sm font-medium text-gray-500">Total Posts</h3>
          <p className="text-3xl font-bold text-gray-900 mt-2">{stats.totalPosts}</p>
        </div>
        <div className="card">
          <h3 className="text-sm font-medium text-gray-500">Avg Engagement</h3>
          <p className="text-3xl font-bold text-gray-900 mt-2">{stats.engagement}%</p>
        </div>
      </div>

      {/* Recent Campaigns */}
      <div className="card">
        <h2 className="text-xl font-bold text-gray-900 mb-4">Recent Campaigns</h2>
        <div className="space-y-4">
          {campaigns.length === 0 ? (
            <p className="text-gray-500 text-center py-8">No campaigns yet. Create your first campaign!</p>
          ) : (
            campaigns.map((campaign: any) => (
              <div
                key={campaign.id}
                className="border border-gray-200 rounded-lg p-4 hover:bg-gray-50 transition-colors"
              >
                <div className="flex justify-between items-start">
                  <div>
                    <h3 className="font-semibold text-gray-900">{campaign.name}</h3>
                    <p className="text-sm text-gray-600 mt-1">{campaign.goal}</p>
                    <div className="flex items-center space-x-4 mt-2">
                      <span className="text-xs px-2 py-1 bg-primary-100 text-primary-800 rounded">
                        {campaign.status}
                      </span>
                      <span className="text-xs text-gray-500">
                        {campaign.target_platforms.join(', ')}
                      </span>
                    </div>
                  </div>
                </div>
              </div>
            ))
          )}
        </div>
      </div>

      {/* AI Agents Status */}
      <div className="card">
        <h2 className="text-xl font-bold text-gray-900 mb-4">AI Agents Status</h2>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          {[
            { name: 'Strategy Agent', status: 'active', icon: '🎯' },
            { name: 'Content Writer', status: 'active', icon: '✍️' },
            { name: 'Creative Agent', status: 'active', icon: '🎨' },
            { name: 'Scheduler Agent', status: 'active', icon: '⏰' },
            { name: 'Posting Agent', status: 'active', icon: '🤖' },
            { name: 'Analytics Agent', status: 'active', icon: '📊' },
            { name: 'Optimization Agent', status: 'active', icon: '🔁' },
          ].map((agent) => (
            <div
              key={agent.name}
              className="border border-gray-200 rounded-lg p-4 text-center"
            >
              <div className="text-2xl mb-2">{agent.icon}</div>
              <div className="text-sm font-medium text-gray-900">{agent.name}</div>
              <div className="text-xs text-green-600 mt-1">{agent.status}</div>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}

