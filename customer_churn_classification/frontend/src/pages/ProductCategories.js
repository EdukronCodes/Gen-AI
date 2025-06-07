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
  BarElement,
  Title,
  Tooltip,
  Legend,
  ArcElement,
} from 'chart.js';
import { Bar, Doughnut } from 'react-chartjs-2';
import RefreshIcon from '@mui/icons-material/Refresh';
import axios from 'axios';

ChartJS.register(
  CategoryScale,
  LinearScale,
  BarElement,
  Title,
  Tooltip,
  Legend,
  ArcElement
);

function ProductCategories() {
  const [loading, setLoading] = useState(true);
  const [categoryStats, setCategoryStats] = useState(null);
  const [churnByCategory, setChurnByCategory] = useState(null);

  useEffect(() => {
    fetchData();
  }, []);

  const fetchData = async () => {
    try {
      setLoading(true);
      const [statsResponse, churnResponse] = await Promise.all([
        axios.get('/api/product-categories/stats/'),
        axios.get('/api/product-categories/churn-by-category/'),
      ]);
      setCategoryStats(statsResponse.data);
      setChurnByCategory(churnResponse.data);
    } catch (error) {
      console.error('Error fetching category data:', error);
    } finally {
      setLoading(false);
    }
  };

  const categoryDistributionData = {
    labels: categoryStats?.categories || [],
    datasets: [
      {
        label: 'Number of Products',
        data: categoryStats?.product_counts || [],
        backgroundColor: 'rgba(54, 162, 235, 0.5)',
        borderColor: 'rgba(54, 162, 235, 1)',
        borderWidth: 1,
      },
    ],
  };

  const churnByCategoryData = {
    labels: churnByCategory?.categories || [],
    datasets: [
      {
        label: 'Churn Rate (%)',
        data: churnByCategory?.churn_rates || [],
        backgroundColor: 'rgba(255, 99, 132, 0.5)',
        borderColor: 'rgba(255, 99, 132, 1)',
        borderWidth: 1,
      },
    ],
  };

  const chartOptions = {
    responsive: true,
    plugins: {
      legend: {
        position: 'top',
      },
      title: {
        display: true,
        text: 'Category Distribution',
      },
    },
  };

  const churnChartOptions = {
    responsive: true,
    plugins: {
      legend: {
        position: 'top',
      },
      title: {
        display: true,
        text: 'Churn Rate by Category',
      },
    },
    scales: {
      y: {
        beginAtZero: true,
        max: 100,
        title: {
          display: true,
          text: 'Churn Rate (%)',
        },
      },
    },
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
        <Typography variant="h4">Product Categories</Typography>
        <Button
          variant="contained"
          startIcon={<RefreshIcon />}
          onClick={fetchData}
        >
          Refresh
        </Button>
      </Box>

      <Grid container spacing={3}>
        <Grid item xs={12} md={6}>
          <Paper sx={{ p: 2 }}>
            <Typography variant="h6" gutterBottom>
              Category Distribution
            </Typography>
            <Box sx={{ height: 400 }}>
              <Bar data={categoryDistributionData} options={chartOptions} />
            </Box>
          </Paper>
        </Grid>

        <Grid item xs={12} md={6}>
          <Paper sx={{ p: 2 }}>
            <Typography variant="h6" gutterBottom>
              Churn Rate by Category
            </Typography>
            <Box sx={{ height: 400 }}>
              <Bar data={churnByCategoryData} options={churnChartOptions} />
            </Box>
          </Paper>
        </Grid>

        <Grid item xs={12}>
          <Paper sx={{ p: 2 }}>
            <Typography variant="h6" gutterBottom>
              Key Insights
            </Typography>
            <Grid container spacing={2}>
              <Grid item xs={12} md={6}>
                <Typography variant="subtitle1" gutterBottom>
                  Top Categories by Product Count:
                </Typography>
                <Box component="ul">
                  {categoryStats?.top_categories.map((category, index) => (
                    <Box component="li" key={index}>
                      {category.name}: {category.count} products
                    </Box>
                  ))}
                </Box>
              </Grid>
              <Grid item xs={12} md={6}>
                <Typography variant="subtitle1" gutterBottom>
                  Categories with Highest Churn:
                </Typography>
                <Box component="ul">
                  {churnByCategory?.top_churn_categories.map((category, index) => (
                    <Box component="li" key={index}>
                      {category.name}: {category.churn_rate.toFixed(1)}% churn rate
                    </Box>
                  ))}
                </Box>
              </Grid>
            </Grid>
          </Paper>
        </Grid>
      </Grid>
    </Box>
  );
}

export default ProductCategories; 