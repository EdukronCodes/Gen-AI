import React, { useState, useEffect } from 'react'
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, PieChart, Pie, Cell } from 'recharts'
import { TrendingUp, Clock, CheckCircle, Ticket } from 'lucide-react'
import axios from 'axios'

const COLORS = ['#0088FE', '#00C49F', '#FFBB28', '#FF8042']

function Dashboard() {
  const [stats, setStats] = useState({
    totalTickets: 0,
    resolvedTickets: 0,
    autoResolved: 0,
    slaCompliance: 0
  })
  const [ticketData, setTicketData] = useState([])
  const [priorityData, setPriorityData] = useState([])

  useEffect(() => {
    fetchDashboardData()
  }, [])

  const fetchDashboardData = async () => {
    try {
      const response = await axios.get('/api/v1/tickets/')
      const tickets = response.data
      
      const resolved = tickets.filter(t => t.status === 'resolved')
      const autoResolved = tickets.filter(t => t.auto_resolved === 'true')
      
      setStats({
        totalTickets: tickets.length,
        resolvedTickets: resolved.length,
        autoResolved: autoResolved.length,
        slaCompliance: 85 // Mock data
      })

      // Prepare chart data
      const categoryCounts = {}
      tickets.forEach(ticket => {
        const cat = ticket.category || 'other'
        categoryCounts[cat] = (categoryCounts[cat] || 0) + 1
      })
      setTicketData(Object.entries(categoryCounts).map(([name, value]) => ({ name, value })))

      const priorityCounts = {}
      tickets.forEach(ticket => {
        const pri = ticket.priority || 'P4'
        priorityCounts[pri] = (priorityCounts[pri] || 0) + 1
      })
      setPriorityData(Object.entries(priorityCounts).map(([name, value]) => ({ name, value })))

    } catch (error) {
      console.error('Error fetching dashboard data:', error)
    }
  }

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
      <h1 className="text-3xl font-bold text-gray-900 mb-8">Dashboard</h1>
      
      {/* Stats Cards */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-6 mb-8">
        <div className="bg-white rounded-lg shadow p-6">
          <div className="flex items-center">
            <Ticket className="h-8 w-8 text-blue-600" />
            <div className="ml-4">
              <p className="text-sm text-gray-600">Total Tickets</p>
              <p className="text-2xl font-bold">{stats.totalTickets}</p>
            </div>
          </div>
        </div>
        
        <div className="bg-white rounded-lg shadow p-6">
          <div className="flex items-center">
            <CheckCircle className="h-8 w-8 text-green-600" />
            <div className="ml-4">
              <p className="text-sm text-gray-600">Resolved</p>
              <p className="text-2xl font-bold">{stats.resolvedTickets}</p>
            </div>
          </div>
        </div>
        
        <div className="bg-white rounded-lg shadow p-6">
          <div className="flex items-center">
            <TrendingUp className="h-8 w-8 text-purple-600" />
            <div className="ml-4">
              <p className="text-sm text-gray-600">Auto-Resolved</p>
              <p className="text-2xl font-bold">{stats.autoResolved}</p>
            </div>
          </div>
        </div>
        
        <div className="bg-white rounded-lg shadow p-6">
          <div className="flex items-center">
            <Clock className="h-8 w-8 text-orange-600" />
            <div className="ml-4">
              <p className="text-sm text-gray-600">SLA Compliance</p>
              <p className="text-2xl font-bold">{stats.slaCompliance}%</p>
            </div>
          </div>
        </div>
      </div>

      {/* Charts */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div className="bg-white rounded-lg shadow p-6">
          <h2 className="text-xl font-bold mb-4">Tickets by Category</h2>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={ticketData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="name" />
              <YAxis />
              <Tooltip />
              <Legend />
              <Bar dataKey="value" fill="#0088FE" />
            </BarChart>
          </ResponsiveContainer>
        </div>

        <div className="bg-white rounded-lg shadow p-6">
          <h2 className="text-xl font-bold mb-4">Tickets by Priority</h2>
          <ResponsiveContainer width="100%" height={300}>
            <PieChart>
              <Pie
                data={priorityData}
                cx="50%"
                cy="50%"
                labelLine={false}
                label={({ name, percent }) => `${name}: ${(percent * 100).toFixed(0)}%`}
                outerRadius={80}
                fill="#8884d8"
                dataKey="value"
              >
                {priorityData.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                ))}
              </Pie>
              <Tooltip />
            </PieChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  )
}

export default Dashboard

