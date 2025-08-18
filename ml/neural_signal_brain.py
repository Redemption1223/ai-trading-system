"""
AGENT_05: Neural Signal Brain
Status: FULLY IMPLEMENTED
Purpose: Neural network for pattern recognition and signal generation with lightweight ML
"""

import logging
import json
import time
import math
import random
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any

# Try to import scientific libraries
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    np = None

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    pd = None

class SimpleNeuralNetwork:
    """Lightweight neural network implementation without heavy dependencies"""
    
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # Initialize weights and biases with small random values
        self.weights_input_hidden = self._initialize_weights(input_size, hidden_size)
        self.weights_hidden_output = self._initialize_weights(hidden_size, output_size)
        self.bias_hidden = [0.0] * hidden_size
        self.bias_output = [0.0] * output_size
        
        # Learning parameters
        self.learning_rate = 0.01
        
    def _initialize_weights(self, rows: int, cols: int) -> List[List[float]]:
        """Initialize weights with small random values"""
        weights = []
        for i in range(rows):
            row = []
            for j in range(cols):
                # Xavier initialization
                limit = math.sqrt(6.0 / (rows + cols))
                row.append(random.uniform(-limit, limit))
            weights.append(row)
        return weights
    
    def _sigmoid(self, x: float) -> float:
        """Sigmoid activation function"""
        try:
            return 1.0 / (1.0 + math.exp(-max(-500, min(500, x))))  # Prevent overflow
        except:
            return 0.5
    
    def _sigmoid_derivative(self, x: float) -> float:
        """Derivative of sigmoid function"""
        s = self._sigmoid(x)
        return s * (1 - s)
    
    def _relu(self, x: float) -> float:
        """ReLU activation function"""
        return max(0, x)
    
    def _tanh(self, x: float) -> float:
        """Hyperbolic tangent activation function"""
        try:
            return math.tanh(x)
        except:
            return 0.0
    
    def forward(self, inputs: List[float]) -> List[float]:
        """Forward pass through the network"""
        if len(inputs) != self.input_size:
            raise ValueError(f"Input size mismatch: expected {self.input_size}, got {len(inputs)}")
        
        # Hidden layer
        hidden = []
        for i in range(self.hidden_size):
            weighted_sum = self.bias_hidden[i]
            for j in range(self.input_size):
                weighted_sum += inputs[j] * self.weights_input_hidden[j][i]
            hidden.append(self._sigmoid(weighted_sum))
        
        # Output layer
        output = []
        for i in range(self.output_size):
            weighted_sum = self.bias_output[i]
            for j in range(self.hidden_size):
                weighted_sum += hidden[j] * self.weights_hidden_output[j][i]
            output.append(self._sigmoid(weighted_sum))
        
        return output
    
    def train_batch(self, training_data: List[Tuple[List[float], List[float]]]) -> float:
        """Train the network on a batch of data"""
        total_error = 0.0
        
        for inputs, targets in training_data:
            # Forward pass
            hidden_inputs = []
            hidden_outputs = []
            
            # Calculate hidden layer
            for i in range(self.hidden_size):
                weighted_sum = self.bias_hidden[i]
                for j in range(self.input_size):
                    weighted_sum += inputs[j] * self.weights_input_hidden[j][i]
                hidden_inputs.append(weighted_sum)
                hidden_outputs.append(self._sigmoid(weighted_sum))
            
            # Calculate output layer
            output_inputs = []
            outputs = []
            for i in range(self.output_size):
                weighted_sum = self.bias_output[i]
                for j in range(self.hidden_size):
                    weighted_sum += hidden_outputs[j] * self.weights_hidden_output[j][i]
                output_inputs.append(weighted_sum)
                outputs.append(self._sigmoid(weighted_sum))
            
            # Calculate error
            output_errors = []
            for i in range(self.output_size):
                error = targets[i] - outputs[i]
                output_errors.append(error)
                total_error += error * error
            
            # Backpropagation
            # Output layer deltas
            output_deltas = []
            for i in range(self.output_size):
                delta = output_errors[i] * self._sigmoid_derivative(output_inputs[i])
                output_deltas.append(delta)
            
            # Hidden layer deltas
            hidden_deltas = []
            for i in range(self.hidden_size):
                error = 0.0
                for j in range(self.output_size):
                    error += output_deltas[j] * self.weights_hidden_output[i][j]
                delta = error * self._sigmoid_derivative(hidden_inputs[i])
                hidden_deltas.append(delta)
            
            # Update weights and biases
            # Update hidden to output weights
            for i in range(self.hidden_size):
                for j in range(self.output_size):
                    self.weights_hidden_output[i][j] += self.learning_rate * output_deltas[j] * hidden_outputs[i]
            
            # Update input to hidden weights
            for i in range(self.input_size):
                for j in range(self.hidden_size):
                    self.weights_input_hidden[i][j] += self.learning_rate * hidden_deltas[j] * inputs[i]
            
            # Update biases
            for i in range(self.output_size):
                self.bias_output[i] += self.learning_rate * output_deltas[i]
            
            for i in range(self.hidden_size):
                self.bias_hidden[i] += self.learning_rate * hidden_deltas[i]
        
        return total_error / len(training_data)

class NeuralSignalBrain:
    """Neural network brain for pattern recognition and signal generation"""
    
    def __init__(self, input_features=20):
        self.name = "NEURAL_SIGNAL_BRAIN"
        self.status = "DISCONNECTED"
        self.version = "1.0.0"
        
        # Network architecture
        self.input_features = input_features  # Technical indicators + price features
        self.hidden_neurons = 32
        self.output_neurons = 3  # [BUY, HOLD, SELL]
        
        # Neural network
        self.network = None
        self.is_trained = False
        self.model_accuracy = 0.0
        self.training_loss = float('inf')
        
        # Training parameters
        self.epochs = 100
        self.batch_size = 32
        self.validation_split = 0.2
        self.min_training_samples = 100
        
        # Feature processing
        self.feature_scaler = None
        self.feature_history = []
        self.max_history = 1000
        
        # Signal generation
        self.prediction_threshold = 0.6  # Minimum confidence for signals
        self.signal_history = []
        self.max_signal_history = 100
        
        # Training data
        self.training_data = []
        self.validation_data = []
        self.last_training_time = None
        self.retrain_interval = 3600  # Retrain every hour
        
        # Performance tracking
        self.predictions_made = 0
        self.correct_predictions = 0
        self.false_positives = 0
        self.false_negatives = 0
        
        # Real-time learning
        self.online_learning = True
        self.learning_thread = None
        self.is_learning = False
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        if not self.logger.handlers:
            logging.basicConfig(
                level=logging.INFO,
                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
    
    def initialize(self):
        """Initialize the neural signal brain"""
        try:
            self.logger.info(f"Initializing {self.name} v{self.version}")
            
            # Create neural network
            self.network = SimpleNeuralNetwork(
                input_size=self.input_features,
                hidden_size=self.hidden_neurons,
                output_size=self.output_neurons
            )
            
            # Initialize feature scaler
            self.feature_scaler = {
                'mean': [0.0] * self.input_features,
                'std': [1.0] * self.input_features
            }
            
            # LIVE DATA ONLY - No synthetic training data
            self.logger.info("Neural Signal Brain initialized for LIVE data training only")
            
            # Perform initial training if we have data
            if len(self.training_data) >= self.min_training_samples:
                initial_loss = self._train_network()
                self.training_loss = initial_loss
                self.is_trained = True
                self.logger.info(f"Initial training completed with loss: {initial_loss:.4f}")
            
            self.status = "INITIALIZED"
            self.logger.info("Neural Signal Brain initialized successfully")
            
            return {
                "status": "initialized",
                "agent": "AGENT_05",
                "network_architecture": f"{self.input_features}-{self.hidden_neurons}-{self.output_neurons}",
                "training_samples": len(self.training_data),
                "is_trained": self.is_trained,
                "initial_loss": self.training_loss if self.is_trained else None,
                "numpy_available": NUMPY_AVAILABLE,
                "pandas_available": PANDAS_AVAILABLE
            }
            
        except Exception as e:
            self.logger.error(f"Initialization failed: {e}")
            self.status = "FAILED"
            return {"status": "failed", "agent": "AGENT_05", "error": str(e)}
    
    # REMOVED: _generate_synthetic_training_data() - LIVE DATA ONLY
    
    def add_training_sample(self, features: List[float], outcome: str):
        """Add a new training sample based on real market outcome"""
        try:
            if len(features) != self.input_features:
                self.logger.warning(f"Feature size mismatch: expected {self.input_features}, got {len(features)}")
                return
            
            # Convert outcome to target vector
            if outcome.upper() == 'BUY':
                target = [1.0, 0.0, 0.0]
            elif outcome.upper() == 'SELL':
                target = [0.0, 0.0, 1.0]
            else:  # HOLD or unknown
                target = [0.0, 1.0, 0.0]
            
            # Scale features
            scaled_features = self._scale_features(features)
            
            # Add to training data
            self.training_data.append((scaled_features, target))
            
            # Limit training data size
            if len(self.training_data) > self.max_history:
                self.training_data = self.training_data[-self.max_history:]
            
            # Trigger retraining if needed
            if (self.online_learning and 
                len(self.training_data) >= self.min_training_samples and
                (self.last_training_time is None or 
                 datetime.now() - self.last_training_time > timedelta(seconds=self.retrain_interval))):
                
                self._schedule_retraining()
            
        except Exception as e:
            self.logger.error(f"Error adding training sample: {e}")
    
    def _scale_features(self, features: List[float]) -> List[float]:
        """Scale features using simple normalization"""
        try:
            if not self.feature_scaler:
                return features
            
            scaled = []
            for i, feature in enumerate(features):
                if i < len(self.feature_scaler['mean']):
                    mean = self.feature_scaler['mean'][i]
                    std = max(self.feature_scaler['std'][i], 1e-8)  # Prevent division by zero
                    scaled_value = (feature - mean) / std
                    # Clamp to prevent extreme values
                    scaled_value = max(-5.0, min(5.0, scaled_value))
                    scaled.append(scaled_value)
                else:
                    scaled.append(feature)
            
            return scaled
            
        except Exception as e:
            self.logger.error(f"Error scaling features: {e}")
            return features
    
    def _update_feature_scaler(self, features_batch: List[List[float]]):
        """Update feature scaler with running statistics"""
        try:
            if not features_batch:
                return
            
            n_features = len(features_batch[0])
            
            # Calculate mean and std for this batch
            for i in range(min(n_features, self.input_features)):
                values = [batch[i] for batch in features_batch if len(batch) > i]
                
                if values:
                    batch_mean = sum(values) / len(values)
                    batch_std = math.sqrt(sum((x - batch_mean) ** 2 for x in values) / len(values)) if len(values) > 1 else 1.0
                    
                    # Update running statistics (simple moving average)
                    alpha = 0.1  # Learning rate for statistics
                    self.feature_scaler['mean'][i] = (1 - alpha) * self.feature_scaler['mean'][i] + alpha * batch_mean
                    self.feature_scaler['std'][i] = (1 - alpha) * self.feature_scaler['std'][i] + alpha * max(batch_std, 0.1)
                    
        except Exception as e:
            self.logger.error(f"Error updating feature scaler: {e}")
    
    def _train_network(self) -> float:
        """Train the neural network"""
        try:
            if not self.network or len(self.training_data) < self.min_training_samples:
                return float('inf')
            
            self.logger.info(f"Training network with {len(self.training_data)} samples")
            
            # Update feature scaler
            features_batch = [sample[0] for sample in self.training_data]
            self._update_feature_scaler(features_batch)
            
            # Train for multiple epochs
            best_loss = float('inf')
            for epoch in range(self.epochs):
                # Shuffle training data
                random.shuffle(self.training_data)
                
                # Train in batches
                epoch_loss = 0.0
                batches = 0
                
                for i in range(0, len(self.training_data), self.batch_size):
                    batch = self.training_data[i:i + self.batch_size]
                    if len(batch) > 0:
                        loss = self.network.train_batch(batch)
                        epoch_loss += loss
                        batches += 1
                
                avg_loss = epoch_loss / max(batches, 1)
                if avg_loss < best_loss:
                    best_loss = avg_loss
                
                # Log progress occasionally
                if epoch % 20 == 0:
                    self.logger.debug(f"Epoch {epoch}: Loss = {avg_loss:.4f}")
            
            # Evaluate on validation data if available
            if self.validation_data:
                self.model_accuracy = self._evaluate_model()
                self.logger.info(f"Training completed. Loss: {best_loss:.4f}, Accuracy: {self.model_accuracy:.4f}")
            
            self.is_trained = True
            self.last_training_time = datetime.now()
            
            return best_loss
            
        except Exception as e:
            self.logger.error(f"Training error: {e}")
            return float('inf')
    
    def _evaluate_model(self) -> float:
        """Evaluate model accuracy on validation data"""
        try:
            if not self.network or not self.validation_data:
                return 0.0
            
            correct = 0
            total = len(self.validation_data)
            
            for features, targets in self.validation_data:
                prediction = self.network.forward(features)
                
                # Get predicted class (highest probability)
                predicted_class = prediction.index(max(prediction))
                actual_class = targets.index(max(targets))
                
                if predicted_class == actual_class:
                    correct += 1
            
            accuracy = correct / total if total > 0 else 0.0
            return accuracy
            
        except Exception as e:
            self.logger.error(f"Error evaluating model: {e}")
            return 0.0
    
    def predict_signal(self, features: List[float]) -> Dict:
        """Predict trading signal from features"""
        try:
            if not self.network or not self.is_trained:
                return {"error": "Model not trained"}
            
            if len(features) != self.input_features:
                return {"error": f"Feature size mismatch: expected {self.input_features}, got {len(features)}"}
            
            # Scale features
            scaled_features = self._scale_features(features)
            
            # Make prediction
            prediction = self.network.forward(scaled_features)
            
            # Interpret prediction
            buy_prob, hold_prob, sell_prob = prediction
            max_prob = max(prediction)
            predicted_action = ['BUY', 'HOLD', 'SELL'][prediction.index(max_prob)]
            
            # Generate signal if confidence is high enough
            signal = None
            if max_prob >= self.prediction_threshold:
                if predicted_action in ['BUY', 'SELL']:
                    signal = {
                        'symbol': 'UNKNOWN',  # Will be set by caller
                        'direction': predicted_action,
                        'confidence': round(max_prob * 100, 1),
                        'strength': round((max_prob - 0.5) * 200, 1),  # Convert to 0-100 scale
                        'probabilities': {
                            'buy': round(buy_prob, 3),
                            'hold': round(hold_prob, 3),
                            'sell': round(sell_prob, 3)
                        },
                        'timestamp': datetime.now().isoformat(),
                        'agent': 'AGENT_05',
                        'model_accuracy': self.model_accuracy,
                        'ai_confidence': round(max_prob * 100, 1),  # For signal coordinator
                        'source': 'neural_network'
                    }
            
            # Update performance tracking
            self.predictions_made += 1
            
            # Add to signal history
            prediction_record = {
                'timestamp': datetime.now().isoformat(),
                'features': features,
                'prediction': prediction,
                'action': predicted_action,
                'confidence': max_prob,
                'signal_generated': signal is not None
            }
            
            self.signal_history.append(prediction_record)
            if len(self.signal_history) > self.max_signal_history:
                self.signal_history = self.signal_history[-self.max_signal_history:]
            
            return {
                "prediction": prediction,
                "action": predicted_action,
                "confidence": max_prob,
                "signal": signal,
                "probabilities": {
                    'buy': buy_prob,
                    'hold': hold_prob,
                    'sell': sell_prob
                }
            }
            
        except Exception as e:
            self.logger.error(f"Prediction error: {e}")
            return {"error": str(e)}
    
    def _schedule_retraining(self):
        """Schedule network retraining in background thread"""
        try:
            if self.is_learning:
                return  # Already retraining
            
            def retrain():
                self.is_learning = True
                try:
                    self.logger.info("Starting background retraining")
                    loss = self._train_network()
                    self.training_loss = loss
                    self.logger.info(f"Background retraining completed with loss: {loss:.4f}")
                except Exception as e:
                    self.logger.error(f"Background retraining error: {e}")
                finally:
                    self.is_learning = False
            
            self.learning_thread = threading.Thread(target=retrain, daemon=True)
            self.learning_thread.start()
            
        except Exception as e:
            self.logger.error(f"Error scheduling retraining: {e}")
    
    def update_model_performance(self, prediction_id: str, actual_outcome: str):
        """Update model performance based on actual market outcome"""
        try:
            # This would be called after a signal result is known
            if actual_outcome.upper() in ['BUY', 'SELL', 'HOLD']:
                # Find the corresponding prediction in history
                for record in reversed(self.signal_history[-10:]):  # Check recent signals
                    if record.get('signal_generated') and record.get('action'):
                        predicted = record['action'].upper()
                        actual = actual_outcome.upper()
                        
                        if predicted == actual:
                            self.correct_predictions += 1
                        elif predicted in ['BUY', 'SELL'] and actual == 'HOLD':
                            self.false_positives += 1
                        elif predicted == 'HOLD' and actual in ['BUY', 'SELL']:
                            self.false_negatives += 1
                        
                        # Add this as training data for continuous learning
                        if 'features' in record:
                            self.add_training_sample(record['features'], actual_outcome)
                        
                        break
            
        except Exception as e:
            self.logger.error(f"Error updating model performance: {e}")
    
    def get_model_architecture(self) -> Dict:
        """Get neural network architecture details"""
        return {
            'input_features': self.input_features,
            'hidden_neurons': self.hidden_neurons,
            'output_neurons': self.output_neurons,
            'total_parameters': (self.input_features * self.hidden_neurons + 
                               self.hidden_neurons * self.output_neurons + 
                               self.hidden_neurons + self.output_neurons),
            'activation_function': 'sigmoid',
            'learning_rate': self.network.learning_rate if self.network else 0.01
        }
    
    def get_training_status(self) -> Dict:
        """Get current training status and metrics"""
        try:
            training_accuracy = 0.0
            if self.predictions_made > 0:
                training_accuracy = (self.correct_predictions / self.predictions_made) * 100
            
            return {
                'is_trained': self.is_trained,
                'is_learning': self.is_learning,
                'training_samples': len(self.training_data),
                'validation_samples': len(self.validation_data),
                'last_training_time': self.last_training_time.isoformat() if self.last_training_time else None,
                'training_loss': self.training_loss,
                'model_accuracy': self.model_accuracy,
                'predictions_made': self.predictions_made,
                'correct_predictions': self.correct_predictions,
                'false_positives': self.false_positives,
                'false_negatives': self.false_negatives,
                'training_accuracy': round(training_accuracy, 2),
                'online_learning_enabled': self.online_learning
            }
            
        except Exception as e:
            self.logger.error(f"Error getting training status: {e}")
            return {"error": str(e)}
    
    def save_model(self, filepath: str) -> bool:
        """Save neural network model to file"""
        try:
            if not self.network:
                return False
            
            model_data = {
                'architecture': self.get_model_architecture(),
                'weights_input_hidden': self.network.weights_input_hidden,
                'weights_hidden_output': self.network.weights_hidden_output,
                'bias_hidden': self.network.bias_hidden,
                'bias_output': self.network.bias_output,
                'feature_scaler': self.feature_scaler,
                'training_status': self.get_training_status(),
                'version': self.version,
                'timestamp': datetime.now().isoformat()
            }
            
            with open(filepath, 'w') as f:
                json.dump(model_data, f, indent=2)
            
            self.logger.info(f"Model saved to {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving model: {e}")
            return False
    
    def load_model(self, filepath: str) -> bool:
        """Load neural network model from file"""
        try:
            with open(filepath, 'r') as f:
                model_data = json.load(f)
            
            # Validate architecture compatibility
            arch = model_data['architecture']
            if (arch['input_features'] != self.input_features or 
                arch['hidden_neurons'] != self.hidden_neurons or
                arch['output_neurons'] != self.output_neurons):
                self.logger.warning("Model architecture mismatch - reinitializing")
                return False
            
            # Load network weights
            self.network.weights_input_hidden = model_data['weights_input_hidden']
            self.network.weights_hidden_output = model_data['weights_hidden_output']
            self.network.bias_hidden = model_data['bias_hidden']
            self.network.bias_output = model_data['bias_output']
            
            # Load feature scaler
            self.feature_scaler = model_data['feature_scaler']
            
            # Load training status
            status = model_data.get('training_status', {})
            self.is_trained = status.get('is_trained', False)
            self.model_accuracy = status.get('model_accuracy', 0.0)
            self.training_loss = status.get('training_loss', float('inf'))
            
            self.logger.info(f"Model loaded from {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading model: {e}")
            return False
    
    def get_feature_importance(self) -> Dict:
        """Calculate approximate feature importance"""
        try:
            if not self.network:
                return {}
            
            # Simple feature importance based on input-hidden layer weights
            importance = []
            
            for i in range(self.input_features):
                # Sum of absolute weights from this input to all hidden neurons
                weight_sum = sum(abs(self.network.weights_input_hidden[i][j]) 
                               for j in range(self.hidden_neurons))
                importance.append(weight_sum)
            
            # Normalize to percentages
            total = sum(importance) if importance else 1.0
            normalized_importance = [(imp / total) * 100 for imp in importance]
            
            # Create feature names (generic for now)
            feature_names = [f"feature_{i+1}" for i in range(self.input_features)]
            
            return {
                'feature_importance': dict(zip(feature_names, normalized_importance)),
                'top_features': sorted(zip(feature_names, normalized_importance), 
                                     key=lambda x: x[1], reverse=True)[:5]
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating feature importance: {e}")
            return {}
    
    def get_performance_metrics(self) -> Dict:
        """Get comprehensive performance metrics"""
        try:
            accuracy = 0.0
            precision = 0.0
            recall = 0.0
            
            if self.predictions_made > 0:
                accuracy = (self.correct_predictions / self.predictions_made) * 100
            
            if (self.correct_predictions + self.false_positives) > 0:
                precision = (self.correct_predictions / (self.correct_predictions + self.false_positives)) * 100
            
            if (self.correct_predictions + self.false_negatives) > 0:
                recall = (self.correct_predictions / (self.correct_predictions + self.false_negatives)) * 100
            
            f1_score = 0.0
            if (precision + recall) > 0:
                f1_score = 2 * (precision * recall) / (precision + recall)
            
            return {
                'predictions_made': self.predictions_made,
                'correct_predictions': self.correct_predictions,
                'false_positives': self.false_positives,
                'false_negatives': self.false_negatives,
                'accuracy': round(accuracy, 2),
                'precision': round(precision, 2),
                'recall': round(recall, 2),
                'f1_score': round(f1_score, 2),
                'model_accuracy': round(self.model_accuracy * 100, 2),
                'training_loss': self.training_loss,
                'is_trained': self.is_trained,
                'is_learning': self.is_learning,
                'signal_history_size': len(self.signal_history),
                'training_data_size': len(self.training_data)
            }
            
        except Exception as e:
            self.logger.error(f"Error getting performance metrics: {e}")
            return {"error": str(e)}
    
    def get_status(self):
        """Get current agent status"""
        return {
            'name': self.name,
            'version': self.version,
            'status': self.status,
            'is_trained': self.is_trained,
            'is_learning': self.is_learning,
            'network_architecture': self.get_model_architecture(),
            'training_samples': len(self.training_data),
            'validation_samples': len(self.validation_data),
            'model_accuracy': self.model_accuracy,
            'training_loss': self.training_loss,
            'predictions_made': self.predictions_made,
            'online_learning': self.online_learning,
            'last_training': self.last_training_time.isoformat() if self.last_training_time else None
        }
    
    def shutdown(self):
        """Clean shutdown of neural signal brain"""
        try:
            self.logger.info("Shutting down Neural Signal Brain...")
            
            # Stop learning thread
            self.is_learning = False
            if self.learning_thread and self.learning_thread.is_alive():
                self.learning_thread.join(timeout=5)
            
            # Save final performance metrics
            metrics = self.get_performance_metrics()
            self.logger.info(f"Final performance metrics: {metrics}")
            
            # Clear memory
            self.training_data.clear()
            self.validation_data.clear()
            self.signal_history.clear()
            self.feature_history.clear()
            
            self.status = "SHUTDOWN"
            self.logger.info("Neural Signal Brain shutdown complete")
            
        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}")

# Agent test
if __name__ == "__main__":
    # Test the neural signal brain
    print("Testing AGENT_05: Neural Signal Brain")
    print("=" * 40)
    
    # Create neural brain
    brain = NeuralSignalBrain(input_features=10)
    result = brain.initialize()
    print(f"Initialization: {result}")
    
    if result['status'] == 'initialized':
        # Test architecture
        arch = brain.get_model_architecture()
        print(f"\nNetwork Architecture: {arch}")
        
        # Test prediction with sample features
        print("\nTesting prediction...")
        sample_features = [0.5, -0.2, 0.8, 0.1, -0.5, 0.3, 0.7, -0.1, 0.4, 0.6]
        prediction = brain.predict_signal(sample_features)
        print(f"Prediction result: {prediction}")
        
        # Test training status
        status = brain.get_training_status()
        print(f"\nTraining status: {status}")
        
        # Test performance metrics
        metrics = brain.get_performance_metrics()
        print(f"\nPerformance metrics: {metrics}")
        
        # Test feature importance
        importance = brain.get_feature_importance()
        print(f"\nFeature importance: {importance}")
        
        # Test adding training sample
        print("\nTesting online learning...")
        brain.add_training_sample(sample_features, 'BUY')
        
        # Test final status
        final_status = brain.get_status()
        print(f"\nFinal status: {final_status}")
        
        # Test shutdown
        print("\nShutting down...")
        brain.shutdown()
        
    print("Neural Signal Brain test completed")