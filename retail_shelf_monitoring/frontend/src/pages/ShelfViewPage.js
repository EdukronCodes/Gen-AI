import React from 'react';
import { Typography, Paper } from '@mui/material';

function ShelfViewPage() {
  return (
    <Paper sx={{ p: 3 }}>
      <Typography variant="h5" gutterBottom>
        Live Shelf Monitoring
      </Typography>
      <Typography>
        This page will display real-time shelf status and video feeds. (To be implemented)
      </Typography>
    </Paper>
  );
}

export default ShelfViewPage; 