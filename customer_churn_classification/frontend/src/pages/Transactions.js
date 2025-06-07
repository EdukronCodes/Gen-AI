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
import ReceiptIcon from '@mui/icons-material/Receipt';
import axios from 'axios';

function Transactions() {
  const [transactions, setTransactions] = useState([]);
  const [loading, setLoading] = useState(true);
  const [selectedTransaction, setSelectedTransaction] = useState(null);

  useEffect(() => {
    fetchTransactions();
  }, []);

  const fetchTransactions = async () => {
    try {
      setLoading(true);
      const response = await axios.get('/api/transactions/');
      setTransactions(response.data.results);
    } catch (error) {
      console.error('Error fetching transactions:', error);
    } finally {
      setLoading(false);
    }
  };

  const getStatusColor = (status) => {
    switch (status) {
      case 'completed':
        return 'success';
      case 'pending':
        return 'warning';
      case 'failed':
        return 'error';
      case 'refunded':
        return 'info';
      default:
        return 'default';
    }
  };

  const columns = [
    { field: 'transaction_id', headerName: 'Transaction ID', width: 180 },
    { field: 'customer_name', headerName: 'Customer', width: 200 },
    {
      field: 'amount',
      headerName: 'Amount',
      width: 130,
      renderCell: (params) => (
        <Typography>${params.value}</Typography>
      ),
    },
    {
      field: 'status',
      headerName: 'Status',
      width: 130,
      renderCell: (params) => (
        <Chip
          label={params.value}
          color={getStatusColor(params.value)}
          size="small"
        />
      ),
    },
    {
      field: 'transaction_date',
      headerName: 'Date',
      width: 180,
      valueFormatter: (params) => {
        return new Date(params.value).toLocaleString();
      },
    },
    {
      field: 'payment_method',
      headerName: 'Payment Method',
      width: 150,
    },
    {
      field: 'product_categories',
      headerName: 'Categories',
      width: 200,
      renderCell: (params) => (
        <Box sx={{ display: 'flex', gap: 0.5 }}>
          {params.value.map((category, index) => (
            <Chip
              key={index}
              label={category}
              size="small"
              variant="outlined"
            />
          ))}
        </Box>
      ),
    },
  ];

  return (
    <Box sx={{ flexGrow: 1, p: 3 }}>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 3 }}>
        <Typography variant="h4">Transactions</Typography>
        <Button
          variant="contained"
          startIcon={<RefreshIcon />}
          onClick={fetchTransactions}
        >
          Refresh
        </Button>
      </Box>

      <Paper sx={{ height: 600, width: '100%' }}>
        <DataGrid
          rows={transactions}
          columns={columns}
          pageSize={10}
          rowsPerPageOptions={[10]}
          checkboxSelection
          disableSelectionOnClick
          loading={loading}
          onRowClick={(params) => setSelectedTransaction(params.row)}
        />
      </Paper>

      {selectedTransaction && (
        <Paper sx={{ mt: 3, p: 2 }}>
          <Typography variant="h6" gutterBottom>
            Transaction Details
          </Typography>
          <Box sx={{ display: 'flex', gap: 2, flexWrap: 'wrap' }}>
            <Chip
              icon={<ReceiptIcon />}
              label={`Transaction ID: ${selectedTransaction.transaction_id}`}
              color="primary"
            />
            <Chip
              label={`Amount: $${selectedTransaction.amount}`}
              color="primary"
            />
            <Chip
              label={`Status: ${selectedTransaction.status}`}
              color={getStatusColor(selectedTransaction.status)}
            />
            <Chip
              label={`Payment Method: ${selectedTransaction.payment_method}`}
              color="primary"
            />
            <Chip
              label={`Date: ${new Date(selectedTransaction.transaction_date).toLocaleString()}`}
              color="primary"
            />
          </Box>
          <Box sx={{ mt: 2 }}>
            <Typography variant="subtitle1" gutterBottom>
              Product Categories:
            </Typography>
            <Box sx={{ display: 'flex', gap: 1, flexWrap: 'wrap' }}>
              {selectedTransaction.product_categories.map((category, index) => (
                <Chip
                  key={index}
                  label={category}
                  color="primary"
                  variant="outlined"
                />
              ))}
            </Box>
          </Box>
        </Paper>
      )}
    </Box>
  );
}

 