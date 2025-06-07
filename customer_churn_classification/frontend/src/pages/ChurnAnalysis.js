import React, { useState, useEffect } from 'react';
import {
  Box,
  Paper,
  Typography,
  Grid,
  Button,
  CircularProgress,
} from '@mui/material';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  Title,
  Tooltip,
  Legend,
} from 'chart.js';
import { Line, Bar } from 'react-chartjs-2';
import axios from 'axios';

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  Title,
  Tooltip,
  Legend
);

function ChurnAnalysis() {
  const [loading, setLoading] = useState(true);
  const [churnStats, setChurnStats] = useState({
    total_customers: 0,
    churned_customers: 0,
    churn_rate: 0,
    average_churn_probability: 0,
    at_risk_customers: 0,
  });
  const [featureImportance, setFeatureImportance] = useState({});

  useEffect(() => {
    fetchData();
  }, []);

  const fetchData = async () => {
    try {
      setLoading(true);
      const [statsResponse, importanceResponse] = await Promise.all([
        axios.get('/api/customers/churn-stats/'),
        axios.get('/api/customers/feature-importance/'),
      ]);
      setChurnStats(statsResponse.data);
      setFeatureImportance(importanceResponse.data);
    } catch (error) {
      console.error('Error fetching analysis data:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleUpdatePredictions = async () => {
    try {
      setLoading(true);
      await axios.post('/api/customers/update-all-churn-probabilities/');
      await fetchData();
    } catch (error) {
      console.error('Error updating predictions:', error);
    } finally {
      setLoading(false);
    }
  };

  const featureImportanceData = {
    labels: Object.keys(featureImportance),
    datasets: [
      {
        label: 'Feature Importance',
        data: Object.values(featureImportance),
        backgroundColor: 'rgba(75, 192, 192, 0.5)',
        borderColor: 'rgb(75, 192, 192)',
        borderWidth: 1,
      },
    ],
  };

  const churnDistributionData = {
    labels: ['Churned', 'Active', 'At Risk'],
    datasets: [
      {
        label: 'Customer Distribution',
        data: [
          churnStats.churned_customers,
          churnStats.total_customers - churnStats.churned_customers - churnStats.at_risk_customers,
          churnStats.at_risk_customers,
        ],
        backgroundColor: [
          'rgba(255, 99, 132, 0.5)',
          'rgba(75, 192, 192, 0.5)',
          'rgba(255, 206, 86, 0.5)',
        ],
        borderColor: [
          'rgb(255, 99, 132)',
          'rgb(75, 192, 192)',
          'rgb(255, 206, 86)',
        ],
        borderWidth: 1,
      },
    ],
  };

  if (loading) {
    return (
      <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '100vh' }}>
        <CircularProgress />
      </Box>
    );
  }

  return (
    <Box sx={{ flexGrow: 1, p: 3 }}>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 3 }}>
        <Typography variant="h4">Churn Analysis</Typography>
        <Button
          variant="contained"
          color="primary"
          onClick={handleUpdatePredictions}
          disabled={loading}
        >
          Update Predictions
        </Button>
      </Box>

      <Grid container spacing={3}>
        {/* Key Metrics */}
        <Grid item xs={12} md={3}>
          <Paper sx={{ p: 2, textAlign: 'center' }}>
            <Typography variant="h6" color="text.secondary">
              Total Customers
            </Typography>
            <Typography variant="h4">
              {churnStats.total_customers}
            </Typography>
          </Paper>
        </Grid>
        <Grid item xs={12} md={3}>
          <Paper sx={{ p: 2, textAlign: 'center' }}>
            <Typography variant="h6" color="error">
              Churn Rate
            </Typography>
            <Typography variant="h4">
              {(churnStats.churn_rate * 100).toFixed(1)}%
            </Typography>
          </Paper>
        </Grid>
        <Grid item xs={12} md={3}>
          <Paper sx={{ p: 2, textAlign: 'center' }}>
            <Typography variant="h6" color="warning">
              At Risk Customers
            </Typography>
            <Typography variant="h4">
              {churnStats.at_risk_customers}
            </Typography>
          </Paper>
        </Grid>
        <Grid item xs={12} md={3}>
          <Paper sx={{ p: 2, textAlign: 'center' }}>
            <Typography variant="h6" color="info">
              Avg. Churn Probability
            </Typography>
            <Typography variant="h4">
              {(churnStats.average_churn_probability * 100).toFixed(1)}%
            </Typography>
          </Paper>
        </Grid>

        {/* Charts */}
        <Grid item xs={12} md={6}>
          <Paper sx={{ p: 2 }}>
            <Typography variant="h6" gutterBottom>
              Customer Distribution
            </Typography>
            <Box sx={{ height: 300 }}>
              <Bar data={churnDistributionData} />
            </Box>
          </Paper>
        </Grid>
        <Grid item xs={12} md={6}>
          <Paper sx={{ p: 2 }}>
            <Typography variant="h6" gutterBottom>
              Feature Importance
            </Typography>
            <Box sx={{ height: 300 }}>
              <Bar data={featureImportanceData} />
            </Box>
          </Paper>
        </Grid>

        {/* Insights */}
        <Grid item xs={12}>
          <Paper sx={{ p: 2 }}>
            <Typography variant="h6" gutterBottom>
              Key Insights
            </Typography>
            <Typography paragraph>
              • {churnStats.churned_customers} customers have churned, representing{' '}
              {(churnStats.churn_rate * 100).toFixed(1)}% of the total customer base.
            </Typography>
            <Typography paragraph>
              • {churnStats.at_risk_customers} customers are at high risk of churning
              (churn probability &gt; 70%).
            </Typography>
            <Typography paragraph>
              • The average churn probability across all customers is{' '}
              {(churnStats.average_churn_probability * 100).toFixed(1)}%.
            </Typography>
            <Typography>
              • Top factors influencing churn: {Object.keys(featureImportance)
                .slice(0, 3)
                .map((feature, index) => (
                  <span key={feature}>
                    {index > 0 ? ', ' : ''}
                    {feature}
                  </span>
                ))}
            </Typography>
          </Paper>
        </Grid>
      </Grid>
    </Box>
  );
}

export default ChurnAnalysis; 