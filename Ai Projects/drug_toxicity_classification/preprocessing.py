"""
Drug Toxicity Classification Data Preprocessing
Handles molecular data preprocessing, feature extraction, and chemical descriptor calculation
"""

import os
import json
import logging
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Chemical informatics imports
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem, MACCSkeys, rdMolDescriptors
from rdkit.Chem.AtomPairs import Pairs, Torsions
from rdkit.Chem.Fingerprints import FingerprintMols
from rdkit.Chem.rdMolDescriptors import GetMorganFingerprintAsBitVect
import networkx as nx

# Machine Learning imports
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
import joblib

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MolecularDataPreprocessor:
    """Preprocessor for molecular data in drug toxicity classification"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.scaler = StandardScaler()
        self.feature_selector = None
        self.pca = None
        self.imputer = SimpleImputer(strategy='mean')
        self.label_encoders = {}
        self.descriptor_names = []
        
    def validate_smiles(self, smiles: str) -> bool:
        """Validate SMILES string"""
        try:
            mol = Chem.MolFromSmiles(smiles)
            return mol is not None
        except:
            return False
    
    def clean_smiles(self, smiles: str) -> str:
        """Clean and standardize SMILES string"""
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                # Convert to canonical SMILES
                cleaned_smiles = Chem.MolToSmiles(mol, canonical=True)
                return cleaned_smiles
            return smiles
        except Exception as e:
            logger.warning(f"Error cleaning SMILES {smiles}: {e}")
            return smiles
    
    def calculate_molecular_descriptors(self, smiles: str) -> Dict[str, float]:
        """Calculate comprehensive molecular descriptors"""
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return {}
            
            descriptors = {}
            
            # Basic descriptors
            descriptors['MolWt'] = Descriptors.MolWt(mol)
            descriptors['LogP'] = Descriptors.MolLogP(mol)
            descriptors['NumHDonors'] = Descriptors.NumHDonors(mol)
            descriptors['NumHAcceptors'] = Descriptors.NumHAcceptors(mol)
            descriptors['NumRotatableBonds'] = Descriptors.NumRotatableBonds(mol)
            descriptors['NumAromaticRings'] = Descriptors.NumAromaticRings(mol)
            descriptors['NumAliphaticRings'] = Descriptors.NumAliphaticRings(mol)
            descriptors['TPSA'] = Descriptors.TPSA(mol)
            descriptors['FractionCsp3'] = Descriptors.FractionCsp3(mol)
            
            # Advanced descriptors
            descriptors['HeavyAtomCount'] = Descriptors.HeavyAtomCount(mol)
            descriptors['RingCount'] = Descriptors.RingCount(mol)
            descriptors['AromaticRings'] = Descriptors.AromaticRings(mol)
            descriptors['SaturatedRings'] = Descriptors.SaturatedRings(mol)
            descriptors['AliphaticRings'] = Descriptors.AliphaticRings(mol)
            descriptors['HeteroatomCount'] = Descriptors.HeteroatomCount(mol)
            
            # Constitutional descriptors
            descriptors['BertzCT'] = Descriptors.BertzCT(mol)
            descriptors['HallKierAlpha'] = Descriptors.HallKierAlpha(mol)
            descriptors['Ipc'] = Descriptors.Ipc(mol)
            
            # Topological descriptors
            descriptors['WienerIndex'] = Descriptors.WienerIndex(mol)
            descriptors['ZagrebIndex'] = Descriptors.ZagrebIndex(mol)
            descriptors['BalabanJ'] = Descriptors.BalabanJ(mol)
            
            # Connectivity descriptors
            descriptors['Chi0'] = Descriptors.Chi0(mol)
            descriptors['Chi1'] = Descriptors.Chi1(mol)
            descriptors['Chi0v'] = Descriptors.Chi0v(mol)
            descriptors['Chi1v'] = Descriptors.Chi1v(mol)
            descriptors['Chi2v'] = Descriptors.Chi2v(mol)
            descriptors['Chi3v'] = Descriptors.Chi3v(mol)
            descriptors['Chi4v'] = Descriptors.Chi4v(mol)
            
            # Kappa shape descriptors
            descriptors['Kappa1'] = Descriptors.Kappa1(mol)
            descriptors['Kappa2'] = Descriptors.Kappa2(mol)
            descriptors['Kappa3'] = Descriptors.Kappa3(mol)
            
            # E-state descriptors
            descriptors['MaxEStateIndex'] = Descriptors.MaxEStateIndex(mol)
            descriptors['MinEStateIndex'] = Descriptors.MinEStateIndex(mol)
            descriptors['MaxPartialCharge'] = Descriptors.MaxPartialCharge(mol)
            descriptors['MinPartialCharge'] = Descriptors.MinPartialCharge(mol)
            
            # VSA descriptors
            vsa_descriptors = Descriptors.VSA_(mol)
            for i, desc in enumerate(vsa_descriptors):
                descriptors[f'VSA_{i}'] = desc
            
            # SlogP_VSA descriptors
            slogp_vsa_descriptors = Descriptors.SlogP_VSA_(mol)
            for i, desc in enumerate(slogp_vsa_descriptors):
                descriptors[f'SlogP_VSA_{i}'] = desc
            
            # SMR_VSA descriptors
            smr_vsa_descriptors = Descriptors.SMR_VSA_(mol)
            for i, desc in enumerate(smr_vsa_descriptors):
                descriptors[f'SMR_VSA_{i}'] = desc
            
            # SlogP_VSA descriptors
            slogp_vsa_descriptors = Descriptors.SlogP_VSA_(mol)
            for i, desc in enumerate(slogp_vsa_descriptors):
                descriptors[f'SlogP_VSA_{i}'] = desc
            
            # PEOE_VSA descriptors
            peoe_vsa_descriptors = Descriptors.PEOE_VSA_(mol)
            for i, desc in enumerate(peoe_vsa_descriptors):
                descriptors[f'PEOE_VSA_{i}'] = desc
            
            return descriptors
            
        except Exception as e:
            logger.error(f"Error calculating descriptors for {smiles}: {e}")
            return {}
    
    def calculate_fingerprints(self, smiles: str) -> Dict[str, List[int]]:
        """Calculate molecular fingerprints"""
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return {}
            
            fingerprints = {}
            
            # Morgan fingerprints (ECFP)
            morgan_fp = GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
            fingerprints['morgan'] = list(morgan_fp.GetOnBits())
            
            # MACCS keys
            maccs_fp = MACCSkeys.GenMACCSKeys(mol)
            fingerprints['maccs'] = list(maccs_fp.GetOnBits())
            
            # Atom pairs
            atom_pairs = Pairs.GetAtomPairFingerprint(mol)
            fingerprints['atom_pairs'] = list(atom_pairs.GetOnBits())
            
            # Topological torsions
            torsions = Torsions.GetTopologicalTorsionFingerprint(mol)
            fingerprints['torsions'] = list(torsions.GetOnBits())
            
            # RDKit fingerprints
            rdkit_fp = FingerprintMols.FingerprintMol(mol)
            fingerprints['rdkit'] = list(rdkit_fp.GetOnBits())
            
            return fingerprints
            
        except Exception as e:
            logger.error(f"Error calculating fingerprints for {smiles}: {e}")
            return {}
    
    def extract_graph_features(self, smiles: str) -> Dict[str, Any]:
        """Extract graph-based features from molecular structure"""
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return {}
            
            # Convert to NetworkX graph
            G = nx.Graph()
            
            # Add nodes (atoms)
            for atom in mol.GetAtoms():
                G.add_node(atom.GetIdx(), 
                          symbol=atom.GetSymbol(),
                          degree=atom.GetDegree(),
                          valence=atom.GetTotalValence(),
                          aromatic=atom.GetIsAromatic())
            
            # Add edges (bonds)
            for bond in mol.GetBonds():
                G.add_edge(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(),
                          bond_type=bond.GetBondType())
            
            # Graph features
            graph_features = {}
            
            # Basic graph metrics
            graph_features['num_nodes'] = G.number_of_nodes()
            graph_features['num_edges'] = G.number_of_edges()
            graph_features['density'] = nx.density(G)
            graph_features['diameter'] = nx.diameter(G) if nx.is_connected(G) else 0
            graph_features['radius'] = nx.radius(G) if nx.is_connected(G) else 0
            
            # Centrality measures
            if nx.is_connected(G):
                graph_features['clustering_coefficient'] = nx.average_clustering(G)
                graph_features['average_shortest_path'] = nx.average_shortest_path_length(G)
            else:
                graph_features['clustering_coefficient'] = 0
                graph_features['average_shortest_path'] = 0
            
            # Degree distribution
            degrees = [d for n, d in G.degree()]
            graph_features['avg_degree'] = np.mean(degrees)
            graph_features['std_degree'] = np.std(degrees)
            graph_features['max_degree'] = max(degrees)
            graph_features['min_degree'] = min(degrees)
            
            # Connectivity
            graph_features['num_connected_components'] = nx.number_connected_components(G)
            graph_features['largest_cc_size'] = len(max(nx.connected_components(G), key=len))
            
            return graph_features
            
        except Exception as e:
            logger.error(f"Error extracting graph features for {smiles}: {e}")
            return {}
    
    def preprocess_molecular_data(self, data: pd.DataFrame, smiles_column: str = 'SMILES') -> Dict[str, Any]:
        """Preprocess molecular data for toxicity classification"""
        try:
            processed_data = {
                'descriptors': [],
                'fingerprints': [],
                'graph_features': [],
                'labels': [],
                'valid_indices': [],
                'feature_names': []
            }
            
            valid_data = []
            
            for idx, row in data.iterrows():
                smiles = row[smiles_column]
                
                # Validate SMILES
                if not self.validate_smiles(smiles):
                    logger.warning(f"Invalid SMILES at index {idx}: {smiles}")
                    continue
                
                # Clean SMILES
                cleaned_smiles = self.clean_smiles(smiles)
                
                # Calculate descriptors
                descriptors = self.calculate_molecular_descriptors(cleaned_smiles)
                if not descriptors:
                    continue
                
                # Calculate fingerprints
                fingerprints = self.calculate_fingerprints(cleaned_smiles)
                
                # Extract graph features
                graph_features = self.extract_graph_features(cleaned_smiles)
                
                # Store data
                processed_data['descriptors'].append(descriptors)
                processed_data['fingerprints'].append(fingerprints)
                processed_data['graph_features'].append(graph_features)
                processed_data['valid_indices'].append(idx)
                
                # Store label if available
                if 'toxicity' in row:
                    processed_data['labels'].append(row['toxicity'])
                elif 'target' in row:
                    processed_data['labels'].append(row['target'])
                
                valid_data.append(row)
            
            # Convert to DataFrame
            descriptors_df = pd.DataFrame(processed_data['descriptors'])
            processed_data['descriptors_df'] = descriptors_df
            processed_data['feature_names'] = descriptors_df.columns.tolist()
            
            # Handle missing values
            processed_data['descriptors_df'] = self.handle_missing_values(descriptors_df)
            
            # Feature scaling
            processed_data['scaled_descriptors'] = self.scale_features(processed_data['descriptors_df'])
            
            # Feature selection
            if len(processed_data['labels']) > 0:
                processed_data['selected_features'] = self.select_features(
                    processed_data['scaled_descriptors'], 
                    processed_data['labels']
                )
            
            logger.info(f"Preprocessed {len(processed_data['valid_indices'])} valid molecules")
            logger.info(f"Extracted {len(processed_data['feature_names'])} molecular descriptors")
            
            return processed_data
            
        except Exception as e:
            logger.error(f"Error preprocessing molecular data: {e}")
            return {}
    
    def handle_missing_values(self, data: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in molecular descriptors"""
        try:
            # Check for missing values
            missing_counts = data.isnull().sum()
            logger.info(f"Missing values per feature: {missing_counts[missing_counts > 0]}")
            
            # Impute missing values
            imputed_data = pd.DataFrame(
                self.imputer.fit_transform(data),
                columns=data.columns,
                index=data.index
            )
            
            return imputed_data
            
        except Exception as e:
            logger.error(f"Error handling missing values: {e}")
            return data
    
    def scale_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Scale molecular descriptors"""
        try:
            scaled_data = pd.DataFrame(
                self.scaler.fit_transform(data),
                columns=data.columns,
                index=data.index
            )
            
            return scaled_data
            
        except Exception as e:
            logger.error(f"Error scaling features: {e}")
            return data
    
    def select_features(self, X: pd.DataFrame, y: List, method: str = 'mutual_info', 
                       k: int = 100) -> pd.DataFrame:
        """Select most important features"""
        try:
            if method == 'mutual_info':
                selector = SelectKBest(score_func=mutual_info_classif, k=min(k, X.shape[1]))
            elif method == 'f_classif':
                selector = SelectKBest(score_func=f_classif, k=min(k, X.shape[1]))
            else:
                logger.warning(f"Unknown feature selection method: {method}")
                return X
            
            selected_features = selector.fit_transform(X, y)
            selected_columns = X.columns[selector.get_support()].tolist()
            
            selected_df = pd.DataFrame(
                selected_features,
                columns=selected_columns,
                index=X.index
            )
            
            logger.info(f"Selected {len(selected_columns)} features using {method}")
            
            return selected_df
            
        except Exception as e:
            logger.error(f"Error selecting features: {e}")
            return X
    
    def apply_pca(self, data: pd.DataFrame, n_components: int = 50) -> pd.DataFrame:
        """Apply PCA for dimensionality reduction"""
        try:
            self.pca = PCA(n_components=min(n_components, data.shape[1]))
            pca_data = self.pca.fit_transform(data)
            
            pca_df = pd.DataFrame(
                pca_data,
                columns=[f'PC_{i+1}' for i in range(pca_data.shape[1])],
                index=data.index
            )
            
            explained_variance = self.pca.explained_variance_ratio_
            logger.info(f"PCA explained variance: {sum(explained_variance):.4f}")
            
            return pca_df
            
        except Exception as e:
            logger.error(f"Error applying PCA: {e}")
            return data
    
    def create_fingerprint_matrix(self, fingerprints: List[Dict[str, List[int]]], 
                                 fp_type: str = 'morgan', n_bits: int = 2048) -> np.ndarray:
        """Create fingerprint matrix from list of fingerprints"""
        try:
            fp_matrix = np.zeros((len(fingerprints), n_bits))
            
            for i, fp_dict in enumerate(fingerprints):
                if fp_type in fp_dict:
                    for bit in fp_dict[fp_type]:
                        if bit < n_bits:
                            fp_matrix[i, bit] = 1
            
            return fp_matrix
            
        except Exception as e:
            logger.error(f"Error creating fingerprint matrix: {e}")
            return np.zeros((len(fingerprints), n_bits))
    
    def preprocess_single_molecule(self, smiles: str) -> Dict[str, Any]:
        """Preprocess a single molecule"""
        try:
            # Validate and clean SMILES
            if not self.validate_smiles(smiles):
                return {}
            
            cleaned_smiles = self.clean_smiles(smiles)
            
            # Calculate features
            descriptors = self.calculate_molecular_descriptors(cleaned_smiles)
            fingerprints = self.calculate_fingerprints(cleaned_smiles)
            graph_features = self.extract_graph_features(cleaned_smiles)
            
            # Convert to DataFrame
            descriptors_df = pd.DataFrame([descriptors])
            
            # Apply preprocessing pipeline
            if hasattr(self.imputer, 'statistics_'):
                descriptors_df = pd.DataFrame(
                    self.imputer.transform(descriptors_df),
                    columns=descriptors_df.columns
                )
            
            if hasattr(self.scaler, 'mean_'):
                descriptors_df = pd.DataFrame(
                    self.scaler.transform(descriptors_df),
                    columns=descriptors_df.columns
                )
            
            return {
                'smiles': cleaned_smiles,
                'descriptors': descriptors_df.iloc[0].to_dict(),
                'fingerprints': fingerprints,
                'graph_features': graph_features,
                'descriptor_vector': descriptors_df.values.flatten()
            }
            
        except Exception as e:
            logger.error(f"Error preprocessing single molecule: {e}")
            return {}
    
    def save_preprocessor(self, filepath: str):
        """Save preprocessor state"""
        try:
            preprocessor_state = {
                'scaler': self.scaler,
                'imputer': self.imputer,
                'pca': self.pca,
                'feature_selector': self.feature_selector,
                'config': self.config,
                'descriptor_names': self.descriptor_names
            }
            
            joblib.dump(preprocessor_state, filepath)
            logger.info(f"Preprocessor saved to {filepath}")
            
        except Exception as e:
            logger.error(f"Error saving preprocessor: {e}")
    
    def load_preprocessor(self, filepath: str):
        """Load preprocessor state"""
        try:
            preprocessor_state = joblib.load(filepath)
            
            self.scaler = preprocessor_state['scaler']
            self.imputer = preprocessor_state['imputer']
            self.pca = preprocessor_state['pca']
            self.feature_selector = preprocessor_state['feature_selector']
            self.config = preprocessor_state['config']
            self.descriptor_names = preprocessor_state['descriptor_names']
            
            logger.info(f"Preprocessor loaded from {filepath}")
            
        except Exception as e:
            logger.error(f"Error loading preprocessor: {e}")

# Example usage
if __name__ == "__main__":
    # Example configuration
    config = {
        'feature_selection_method': 'mutual_info',
        'n_features': 100,
        'pca_components': 50,
        'fingerprint_bits': 2048
    }
    
    preprocessor = MolecularDataPreprocessor(config)
    
    # Example data
    sample_data = pd.DataFrame({
        'SMILES': [
            'CC(=O)OC1=CC=CC=C1C(=O)O',  # Aspirin
            'CC(C)CC1=CC=C(C=C1)C(C)C(=O)O',  # Ibuprofen
            'CC1=C(C(=CC=C1)NC(=O)C2=CC=C(C=C2)CN3CCN(CC3)C)NC4=NC=CC(=N4)C5=CN=CC=C5'  # Complex molecule
        ],
        'toxicity': [0, 0, 1]
    })
    
    # Preprocess data
    processed = preprocessor.preprocess_molecular_data(sample_data)
    
    print(f"Processed {len(processed['valid_indices'])} molecules")
    print(f"Extracted {len(processed['feature_names'])} descriptors")
    
    # Preprocess single molecule
    single_mol = preprocessor.preprocess_single_molecule('CC(=O)OC1=CC=CC=C1C(=O)O')
    print(f"Single molecule descriptors: {len(single_mol.get('descriptors', {}))}") 