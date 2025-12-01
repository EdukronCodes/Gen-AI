'use client'

import { useState, useEffect } from 'react'
import axios from 'axios'

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'

const platforms = [
  { name: 'Instagram', id: 'instagram', icon: '📷', color: 'bg-gradient-to-r from-purple-500 to-pink-500' },
  { name: 'Facebook', id: 'facebook', icon: '👥', color: 'bg-gradient-to-r from-blue-500 to-blue-700' },
  { name: 'Twitter/X', id: 'twitter', icon: '🐦', color: 'bg-gradient-to-r from-black to-gray-800' },
  { name: 'YouTube', id: 'youtube', icon: '▶️', color: 'bg-gradient-to-r from-red-500 to-red-700' },
]

export default function PlatformConnections() {
  const [connections, setConnections] = useState<any[]>([])
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    fetchConnections()
  }, [])

  const fetchConnections = async () => {
    try {
      const response = await axios.get(`${API_URL}/api/v1/platforms`)
      setConnections(response.data)
    } catch (error) {
      console.error('Error fetching connections:', error)
    } finally {
      setLoading(false)
    }
  }

  const handleConnect = async (platformId: string) => {
    const accessToken = prompt(`Enter ${platformId} access token:`)
    if (!accessToken) return

    try {
      await axios.post(`${API_URL}/api/v1/platforms/connect`, {
        user_id: 1,
        platform: platformId,
        access_token: accessToken,
        platform_username: prompt(`Enter ${platformId} username:`) || ''
      })
      
      fetchConnections()
      alert('Platform connected successfully!')
    } catch (error: any) {
      alert('Error connecting platform: ' + (error.response?.data?.detail || error.message))
    }
  }

  const handleDisconnect = async (connectionId: number) => {
    if (!confirm('Are you sure you want to disconnect this platform?')) return

    try {
      await axios.delete(`${API_URL}/api/v1/platforms/${connectionId}`)
      fetchConnections()
      alert('Platform disconnected')
    } catch (error: any) {
      alert('Error disconnecting platform: ' + (error.response?.data?.detail || error.message))
    }
  }

  if (loading) {
    return <div className="text-center py-12">Loading...</div>
  }

  return (
    <div>
      <div className="card mb-6">
        <h2 className="text-2xl font-bold text-gray-900 mb-4">Platform Connections</h2>
        <p className="text-gray-600">
          Connect your social media accounts to enable automated posting
        </p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {platforms.map((platform) => {
          const connection = connections.find(c => c.platform === platform.id)
          const isConnected = connection && connection.is_active

          return (
            <div key={platform.id} className="card">
              <div className="flex items-start justify-between">
                <div className="flex items-center space-x-4">
                  <div className={`w-16 h-16 ${platform.color} rounded-lg flex items-center justify-center text-2xl`}>
                    {platform.icon}
                  </div>
                  <div>
                    <h3 className="text-xl font-semibold text-gray-900">{platform.name}</h3>
                    {isConnected ? (
                      <p className="text-sm text-green-600 mt-1">
                        Connected as @{connection.platform_username || 'user'}
                      </p>
                    ) : (
                      <p className="text-sm text-gray-500 mt-1">Not connected</p>
                    )}
                  </div>
                </div>
              </div>

              <div className="mt-4">
                {isConnected ? (
                  <button
                    onClick={() => handleDisconnect(connection.id)}
                    className="btn-secondary w-full"
                  >
                    Disconnect
                  </button>
                ) : (
                  <button
                    onClick={() => handleConnect(platform.id)}
                    className="btn-primary w-full"
                  >
                    Connect Platform
                  </button>
                )}
              </div>

              <div className="mt-4 p-3 bg-gray-50 rounded-lg">
                <p className="text-xs text-gray-600">
                  <strong>Features:</strong> Automated posting, scheduling, analytics tracking
                </p>
              </div>
            </div>
          )
        })}
      </div>

      <div className="mt-6 card bg-yellow-50 border border-yellow-200">
        <h3 className="font-semibold text-yellow-800 mb-2">⚠️ API Setup Required</h3>
        <p className="text-sm text-yellow-700">
          To connect platforms, you'll need to:
          <ol className="list-decimal list-inside mt-2 space-y-1">
            <li>Create developer accounts on each platform</li>
            <li>Set up OAuth applications</li>
            <li>Generate access tokens</li>
            <li>Add tokens in your backend .env file</li>
          </ol>
          <p className="mt-2">
            For demo purposes, you can use placeholder tokens.
          </p>
        </p>
      </div>
    </div>
  )
}

