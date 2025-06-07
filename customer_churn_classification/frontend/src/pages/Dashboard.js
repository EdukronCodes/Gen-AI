import React, { useState, useEffect } from 'react';
import { Grid, Paper, Typography, Box } from '@mui/material';
import { Line, Doughnut } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  ArcElement,
} from 'chart.js';
import axios from 'axios';

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  ArcElement
);

function Dashboard() {
  const [churnStats, setChurnStats] = useState({
    total_customers: 0,
    churned_customers: 0,
    churn_rate: 0,
    average_churn_probability: 0,
    at_risk_customers: 0,
  });

  const [featureImportance, setFeatureImportance] = useState({});

  useEffect(() => {
    const fetchData = async () => {
      try {
        const [statsResponse, importanceResponse] = await Promise.all([
          axios.get('/api/customers/churn-stats/'),
          axios.get('/api/customers/feature-importance/'),
        ]);
        setChurnStats(statsResponse.data);
        setFeatureImportance(importanceResponse.data);
      } catch (error) {
        console.error('Error fetching dashboard data:', error);
      }
    };

    fetchData();
  }, []);

  const churnRateData = {
    labels: ['Churned', 'Active', 'At Risk'],
    datasets: [
      {
        data: [
          churnStats.churned_customers,
          churnStats.total_customers - churnStats.churned_customers - churnStats.at_risk_customers,
          churnStats.at_risk_customers,
        ],
        backgroundColor: ['#dc3545', '#28a745', '#ffc107'],
      },
    ],
  };

  const featureImportanceData = {
    labels: Object.keys(featureImportance),
    datasets: [
      {
        label: 'Feature Importance',
        data: Object.values(featureImportance),
        fill: false,
        borderColor: 'rgb(75, 192, 192)',
        tension: 0.1,
      },
    ],
  };

  return (
    <Box sx={{ flexGrow: 1, p: 3 }}>
      <Typography variant="h4" gutterBottom>
        Dashboard
      </Typography>
      <Grid container spacing={3}>
        {/* Key Metrics */}
        <Grid item xs={12} md={3}>
          <Paper sx={{ p: 2, display: 'flex', flexDirection: 'column' }}>
            <Typography component="h2" variant="h6" color="primary" gutterBottom>
              Total Customers
            </Typography>
            <Typography component="p" variant="h4">
              {churnStats.total_customers}
            </Typography>
          </Paper>
        </Grid>
        <Grid item xs={12} md={3}>
          <Paper sx={{ p: 2, display: 'flex', flexDirection: 'column' }}>
            <Typography component="h2" variant="h6" color="error" gutterBottom>
              Churned Customers
            </Typography>
            <Typography component="p" variant="h4">
              {churnStats.churned_customers}
            </Typography>
          </Paper>
        </Grid>
        <Grid item xs={12} md={3}>
          <Paper sx={{ p: 2, display: 'flex', flexDirection: 'column' }}>
            <Typography component="h2" variant="h6" color="warning" gutterBottom>
              At Risk Customers
            </Typography>
            <Typography component="p" variant="h4">
              {churnStats.at_risk_customers}
            </Typography>
          </Paper>
        </Grid>
        <Grid item xs={12} md={3}>
          <Paper sx={{ p: 2, display: 'flex', flexDirection: 'column' }}>
            <Typography component="h2" variant="h6" color="info" gutterBottom>
              Churn Rate
            </Typography>
            <Typography component="p" variant="h4">
              {(churnStats.churn_rate * 100).toFixed(1)}%
            </Typography>
          </Paper>
        </Grid>

        {/* Charts */}
        <Grid item xs={12} md={6}>
          <Paper sx={{ p: 2, display: 'flex', flexDirection: 'column' }}>
            <Typography component="h2" variant="h6" color="primary" gutterBottom>
              Customer Distribution
            </Typography>
            <Box sx={{ height: 300 }}>
              <Doughnut data={churnRateData} />
            </Box>
          </Paper>
        </Grid>
        <Grid item xs={12} md={6}>
          <Paper sx={{ p: 2, display: 'flex', flexDirection: 'column' }}>
            <Typography component="h2" variant="h6" color="primary" gutterBottom>
              Feature Importance
            </Typography>
            <Box sx={{ height: 300 }}>
              <Line data={featureImportanceData} />
            </Box>
          </Paper>
        </Grid>
      </Grid>
    </Box>
  );
}

export default Dashboard; 