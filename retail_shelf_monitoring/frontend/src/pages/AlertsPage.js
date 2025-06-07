import React from 'react';
import { Typography, Paper } from '@mui/material';

function AlertsPage() {
  return (
    <Paper sx={{ p: 3 }}>
      <Typography variant="h5" gutterBottom>
        Alerts & Notifications
      </Typography>
      <Typography>
        This page will show alerts for out-of-stock, misplaced, or low-stock products. (To be implemented)
      </Typography>
    </Paper>
  );
}

export default AlertsPage; 