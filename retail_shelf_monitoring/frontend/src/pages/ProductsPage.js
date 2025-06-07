import React from 'react';
import { Typography, Paper } from '@mui/material';

function ProductsPage() {
  return (
    <Paper sx={{ p: 3 }}>
      <Typography variant="h5" gutterBottom>
        Product Management
      </Typography>
      <Typography>
        Here you can add, edit, and view products. (To be implemented)
      </Typography>
    </Paper>
  );
}

export default ProductsPage; 