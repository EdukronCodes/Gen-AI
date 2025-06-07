import React, { useState, useEffect } from 'react';
import {
  Box,
  Paper,
  Typography,
  Button,
  Chip,
  IconButton,
  Tooltip,
} from '@mui/material';
import { DataGrid } from '@mui/x-data-grid';
import RefreshIcon from '@mui/icons-material/Refresh';
import WarningIcon from '@mui/icons-material/Warning';
import axios from 'axios';

function Customers() {
  const [customers, setCustomers] = useState([]);
  const [loading, setLoading] = useState(true);
  const [selectedCustomer, setSelectedCustomer] = useState(null);

  useEffect(() => {
    fetchCustomers();
  }, []);

  const fetchCustomers = async () => {
    try {
      setLoading(true);
      const response = await axios.get('/api/customers/');
      setCustomers(response.data.results);
    } catch (error) {
      console.error('Error fetching customers:', error);
    } finally {
      setLoading(false);
    }
  };

  const handlePredictChurn = async (customerId) => {
    try {
      const response = await axios.post(`/api/customers/${customerId}/predict-churn/`);
      const updatedCustomer = response.data;
      setCustomers(customers.map(customer =>
        customer.id === customerId
          ? { ...customer, churn_probability: updatedCustomer.churn_probability, is_churned: updatedCustomer.is_churned }
          : customer
      ));
    } catch (error) {
      console.error('Error predicting churn:', error);
    }
  };

  const columns = [
    { field: 'customer_id', headerName: 'Customer ID', width: 130 },
    { field: 'first_name', headerName: 'First Name', width: 130 },
    { field: 'last_name', headerName: 'Last Name', width: 130 },
    { field: 'email', headerName: 'Email', width: 200 },
    {
      field: 'churn_probability',
      headerName: 'Churn Probability',
      width: 150,
      renderCell: (params) => (
        <Chip
          label={`${(params.value * 100).toFixed(1)}%`}
          color={
            params.value >= 0.7
              ? 'error'
              : params.value >= 0.4
              ? 'warning'
              : 'success'
          }
        />
      ),
    },
    {
      field: 'is_churned',
      headerName: 'Status',
      width: 120,
      renderCell: (params) => (
        <Chip
          label={params.value ? 'Churned' : 'Active'}
          color={params.value ? 'error' : 'success'}
        />
      ),
    },
    {
      field: 'actions',
      headerName: 'Actions',
      width: 120,
      renderCell: (params) => (
        <Tooltip title="Predict Churn">
          <IconButton
            onClick={() => handlePredictChurn(params.row.id)}
            color="primary"
          >
            <RefreshIcon />
          </IconButton>
        </Tooltip>
      ),
    },
  ];

  return (
    <Box sx={{ flexGrow: 1, p: 3 }}>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 3 }}>
        <Typography variant="h4">Customers</Typography>
        <Button
          variant="contained"
          startIcon={<RefreshIcon />}
          onClick={fetchCustomers}
        >
          Refresh
        </Button>
      </Box>

      <Paper sx={{ height: 600, width: '100%' }}>
        <DataGrid
          rows={customers}
          columns={columns}
          pageSize={10}
          rowsPerPageOptions={[10]}
          checkboxSelection
          disableSelectionOnClick
          loading={loading}
          onRowClick={(params) => setSelectedCustomer(params.row)}
        />
      </Paper>

      {selectedCustomer && (
        <Paper sx={{ mt: 3, p: 2 }}>
          <Typography variant="h6" gutterBottom>
            Customer Details
          </Typography>
          <Box sx={{ display: 'flex', gap: 2, flexWrap: 'wrap' }}>
            <Chip
              icon={<WarningIcon />}
              label={`Churn Risk: ${(selectedCustomer.churn_probability * 100).toFixed(1)}%`}
              color={
                selectedCustomer.churn_probability >= 0.7
                  ? 'error'
                  : selectedCustomer.churn_probability >= 0.4
                  ? 'warning'
                  : 'success'
              }
            />
            <Chip
              label={`Total Purchases: ${selectedCustomer.total_purchases}`}
              color="primary"
            />
            <Chip
              label={`Total Spent: $${selectedCustomer.total_spent}`}
              color="primary"
            />
            <Chip
              label={`Average Order Value: $${selectedCustomer.average_order_value}`}
              color="primary"
            />
            <Chip
              label={`Days Since Last Purchase: ${selectedCustomer.days_since_last_purchase}`}
              color="primary"
            />
          </Box>
        </Paper>
      )}
    </Box>
  );
}

export default Customers; 