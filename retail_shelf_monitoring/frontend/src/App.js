import React from 'react';
import { Routes, Route, Navigate } from 'react-router-dom';
import { CssBaseline, Box, AppBar, Toolbar, Typography, Container } from '@mui/material';
import CameraPage from './pages/CameraPage';
import ShelfViewPage from './pages/ShelfViewPage';
import AlertsPage from './pages/AlertsPage';
import ProductsPage from './pages/ProductsPage';
import Sidebar from './components/Sidebar';

function App() {
  return (
    <Box sx={{ display: 'flex' }}>
      <CssBaseline />
      <Sidebar />
      <Box component="main" sx={{ flexGrow: 1, p: 3 }}>
        <AppBar position="fixed" sx={{ zIndex: (theme) => theme.zIndex.drawer + 1 }}>
          <Toolbar>
            <Typography variant="h6" noWrap component="div">
              Retail Shelf Monitoring
            </Typography>
          </Toolbar>
        </AppBar>
        <Toolbar />
        <Container maxWidth="lg" sx={{ mt: 4, mb: 4 }}>
          <Routes>
            <Route path="/" element={<Navigate to="/shelf" />} />
            <Route path="/shelf" element={<ShelfViewPage />} />
            <Route path="/cameras" element={<CameraPage />} />
            <Route path="/alerts" element={<AlertsPage />} />
            <Route path="/products" element={<ProductsPage />} />
          </Routes>
        </Container>
      </Box>
    </Box>
  );
}

export default App; 