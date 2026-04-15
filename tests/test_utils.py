"""Tests for utility modules."""

import unittest
import numpy as np
import os
import tempfile
from keraformer.utils import (
    # Weight initialization
    xavier_uniform,
    xavier_normal,
    he_uniform,
    he_normal,
    normal,
    uniform,
    zeros,
    ones,
    orthogonal,
    # Inference
    greedy_decode,
    beam_search,
    temperature_sampling,
    top_k_sampling,
    top_p_sampling,
    # Checkpointing
    save_checkpoint,
    load_checkpoint,
    get_checkpoint_info,
    compare_checkpoints,
    # Data loading
    Dataset,
    DataLoader,
    pad_sequences,
    create_autoregressive_dataset,
    create_mask_for_padding,
    create_causal_mask,
    MetricsTracker,
    accuracy,
    perplexity,
    bleu_score,
    top_k_accuracy,
    # Visualization
    plot_attention_heads,
    plot_embeddings,
    plot_loss_curve,
    plot_gradient_flow,
    plot_token_distribution,
    compare_attention_patterns,
)


class TestWeightInitializers(unittest.TestCase):
    """Test weight initialization functions."""
    
    def test_xavier_uniform_shape(self):
        """Test Xavier uniform initialization shape."""
        shape = (100, 200)
        weights = xavier_uniform(shape)
        self.assertEqual(weights.shape, shape)
        self.assertTrue(np.all(np.isfinite(weights)))
    
    def test_xavier_uniform_bounds(self):
        """Test Xavier uniform initialization bounds."""
        shape = (1000, 1000)
        weights = xavier_uniform(shape)
        limit = np.sqrt(6 / (shape[0] + shape[1]))
        self.assertTrue(np.all(weights >= -limit * 1.1))
        self.assertTrue(np.all(weights <= limit * 1.1))
    
    def test_xavier_normal_shape(self):
        """Test Xavier normal initialization shape."""
        shape = (100, 200)
        weights = xavier_normal(shape)
        self.assertEqual(weights.shape, shape)
    
    def test_he_uniform_shape(self):
        """Test He uniform initialization shape."""
        shape = (100, 200)
        weights = he_uniform(shape)
        self.assertEqual(weights.shape, shape)
    
    def test_he_normal_shape(self):
        """Test He normal initialization shape."""
        shape = (100, 200)
        weights = he_normal(shape)
        self.assertEqual(weights.shape, shape)
    
    def test_normal_shape_and_stats(self):
        """Test normal initialization shape and statistics."""
        shape = (10000,)
        weights = normal(shape, mean=0, std=1)
        self.assertEqual(weights.shape, shape)
        # Check empirical mean and std are close
        self.assertAlmostEqual(np.mean(weights), 0, delta=0.1)
        self.assertAlmostEqual(np.std(weights), 1, delta=0.1)
    
    def test_uniform_shape_and_bounds(self):
        """Test uniform initialization shape and bounds."""
        shape = (1000,)
        weights = uniform(shape, low=-1, high=1)
        self.assertEqual(weights.shape, shape)
        self.assertTrue(np.all(weights >= -1))
        self.assertTrue(np.all(weights <= 1))
    
    def test_zeros_and_ones(self):
        """Test zeros and ones initialization."""
        shape = (10, 20)
        z = zeros(shape)
        o = ones(shape)
        self.assertTrue(np.all(z == 0))
        self.assertTrue(np.all(o == 1))
    
    def test_orthogonal_initialization(self):
        """Test orthogonal matrix initialization."""
        shape = (100, 100)
        weights = orthogonal(shape)
        self.assertEqual(weights.shape, shape)
        
        # Check orthogonality: W @ W.T ≈ I
        product = weights @ weights.T
        identity = np.eye(shape[0])
        np.testing.assert_allclose(product, identity, atol=1e-5)


class TestInference(unittest.TestCase):
    """Test inference utilities."""
    
    def test_greedy_decode_shape(self):
        """Test greedy decode output shape."""
        batch_size, vocab_size = 4, 1000
        logits = np.random.randn(batch_size, vocab_size)
        next_tokens = greedy_decode(logits, sequence_length=10)
        self.assertEqual(next_tokens.size, batch_size)
    
    def test_beam_search_output(self):
        """Test beam search output."""
        vocab_size = 1000
        logits = np.random.randn(vocab_size)
        beam_width = 5
        sequences, scores = beam_search(logits, beam_width=beam_width)
        self.assertEqual(len(sequences), beam_width)
        self.assertEqual(len(scores), beam_width)
    
    def test_temperature_sampling_shape(self):
        """Test temperature sampling shape."""
        vocab_size = 1000
        logits = np.random.randn(vocab_size)
        samples = temperature_sampling(logits, temperature=1.0, num_samples=10)
        self.assertEqual(len(samples), 10)
        self.assertTrue(np.all(samples >= 0))
        self.assertTrue(np.all(samples < vocab_size))
    
    def test_top_k_sampling_validity(self):
        """Test top-k sampling produces valid tokens."""
        vocab_size = 1000
        logits = np.random.randn(vocab_size)
        samples = top_k_sampling(logits, k=50, temperature=1.0)
        self.assertTrue(np.all(samples >= 0))
        self.assertTrue(np.all(samples < vocab_size))
    
    def test_top_p_sampling_validity(self):
        """Test top-p sampling produces valid tokens."""
        vocab_size = 1000
        logits = np.random.randn(vocab_size)
        samples = top_p_sampling(logits, p=0.9, temperature=1.0)
        self.assertTrue(np.all(samples >= 0))
        self.assertTrue(np.all(samples < vocab_size))


class TestCheckpointing(unittest.TestCase):
    """Test checkpoint saving and loading."""
    
    def setUp(self):
        """Create temporary directory for checkpoints."""
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up temporary directory."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_save_and_load_weights(self):
        """Test saving and loading weights."""
        weights = {
            'layer1': np.random.randn(10, 20),
            'layer2': np.random.randn(20, 30),
        }
        
        path = os.path.join(self.temp_dir, 'checkpoint.npz')
        save_checkpoint(path, weights, step=100)
        
        loaded = load_checkpoint(path, return_optimizer_state=False, return_metadata=False)
        
        self.assertIn('weights', loaded)
        self.assertEqual(set(loaded['weights'].keys()), set(weights.keys()))
        np.testing.assert_array_equal(loaded['weights']['layer1'], weights['layer1'])
    
    def test_save_and_load_with_optimizer_state(self):
        """Test saving and loading with optimizer state."""
        weights = {'layer1': np.random.randn(10, 20)}
        optimizer_state = {'m': np.zeros(10), 'v': np.zeros(10)}
        
        path = os.path.join(self.temp_dir, 'checkpoint.npz')
        save_checkpoint(path, weights, optimizer_state=optimizer_state, step=50)
        
        loaded = load_checkpoint(path, return_optimizer_state=True)
        
        self.assertIn('optimizer_state', loaded)
        self.assertEqual(set(loaded['optimizer_state'].keys()), set(optimizer_state.keys()))
    
    def test_checkpoint_info(self):
        """Test getting checkpoint info without loading full weights."""
        weights = {'layer1': np.random.randn(100, 200)}
        
        path = os.path.join(self.temp_dir, 'checkpoint.npz')
        save_checkpoint(path, weights, step=75)
        
        info = get_checkpoint_info(path)
        
        self.assertEqual(info['step'], 75)
        self.assertIn('layer1', info['weight_names'])
        self.assertEqual(info['weight_shapes'][0], (100, 200))
    
    def test_compare_checkpoints(self):
        """Test comparing two checkpoints."""
        weights1 = {'layer1': np.ones((10, 10))}
        weights2 = {'layer1': np.ones((10, 10)) * 2}
        
        path1 = os.path.join(self.temp_dir, 'ckpt1.npz')
        path2 = os.path.join(self.temp_dir, 'ckpt2.npz')
        
        save_checkpoint(path1, weights1, step=10)
        save_checkpoint(path2, weights2, step=20)
        
        comparison = compare_checkpoints(path1, path2)
        
        self.assertEqual(comparison['step_diff'], 10)
        self.assertIn('layer1', comparison['different_weights'])


class TestDataLoading(unittest.TestCase):
    """Test data loading utilities."""
    
    def test_dataset_creation(self):
        """Test Dataset class creation."""
        sequences = np.random.randint(0, 100, (32, 50))
        labels = np.random.randint(0, 10, (32,))
        
        dataset = Dataset(sequences, labels=labels)
        
        self.assertEqual(len(dataset), 32)
        seq, label = dataset[0]
        self.assertEqual(seq.shape, (50,))
    
    def test_dataloader_batching(self):
        """Test DataLoader batching."""
        sequences = np.random.randint(0, 100, (100, 50))
        dataset = Dataset(sequences)
        
        dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
        
        batches = list(dataloader)
        self.assertEqual(len(batches), 4)  # 100 / 32 rounded up
    
    def test_pad_sequences(self):
        """Test sequence padding."""
        sequences = [
            np.array([1, 2, 3]),
            np.array([4, 5, 6, 7, 8]),
            np.array([9, 10]),
        ]
        
        padded = pad_sequences(sequences, max_length=6, pad_value=0)
        
        self.assertEqual(padded.shape, (3, 6))
        self.assertEqual(padded[0, 3], 0)  # Padded value
    
    def test_create_autoregressive_dataset(self):
        """Test creating autoregressive pairs."""
        sequences = np.array([[1, 2, 3, 4, 5, 6]])
        
        inputs, targets = create_autoregressive_dataset(sequences, target_length=1)
        
        self.assertEqual(len(inputs), 5)
        self.assertEqual(len(targets), 5)
    
    def test_create_mask_for_padding(self):
        """Test creating padding mask."""
        sequences = np.array([
            [1, 2, 3, 0, 0],
            [5, 6, 0, 0, 0],
        ])
        
        mask = create_mask_for_padding(sequences, pad_token_id=0)
        
        np.testing.assert_array_equal(mask[0], [1, 1, 1, 0, 0])
        np.testing.assert_array_equal(mask[1], [1, 1, 0, 0, 0])
    
    def test_create_causal_mask(self):
        """Test creating causal mask."""
        seq_len = 5
        mask = create_causal_mask(seq_len)
        
        self.assertEqual(mask.shape, (5, 5))
        self.assertTrue(np.all(mask == np.tril(np.ones(5))))


class TestMetrics(unittest.TestCase):
    """Test metrics tracking and computation."""
    
    def test_metrics_tracker(self):
        """Test MetricsTracker functionality."""
        tracker = MetricsTracker(use_mlflow=False)
        
        tracker.update(step=0, loss=0.5, accuracy=0.9)
        tracker.update(step=1, loss=0.4, accuracy=0.92)
        
        self.assertAlmostEqual(tracker.get_latest('loss'), 0.4, places=5)
        self.assertEqual(len(tracker.get_metric('accuracy')), 2)
    
    def test_accuracy_computation(self):
        """Test accuracy metric."""
        predictions = np.array([[1, 2, 3], [4, 5, 6]])
        targets = np.array([[1, 2, 4], [4, 5, 6]])
        
        acc = accuracy(predictions, targets)
        
        self.assertAlmostEqual(acc, 5/6, places=5)
    
    def test_accuracy_with_mask(self):
        """Test accuracy with masking."""
        predictions = np.array([[1, 2, 3]])
        targets = np.array([[1, 2, 4]])
        mask = np.array([[1, 1, 0]])
        
        acc = accuracy(predictions, targets, mask=mask)
        
        self.assertEqual(acc, 1.0)
    
    def test_perplexity_computation(self):
        """Test perplexity metric."""
        batch_size, seq_len, vocab_size = 2, 5, 100
        logits = np.random.randn(batch_size, seq_len, vocab_size)
        targets = np.random.randint(0, vocab_size, (batch_size, seq_len))
        
        ppl = perplexity(logits, targets)
        
        self.assertGreater(ppl, 0)
        self.assertTrue(np.isfinite(ppl))
    
    def test_bleu_score_computation(self):
        """Test BLEU score."""
        predictions = [[1, 2, 3, 4, 5]]
        references = [[1, 2, 3, 4, 5]]
        
        bleu = bleu_score(predictions, references)
        
        self.assertAlmostEqual(bleu, 1.0, places=5)
    
    def test_top_k_accuracy(self):
        """Test top-k accuracy."""
        batch_size, num_classes = 10, 100
        predictions = np.random.randn(batch_size, num_classes)
        targets = np.random.randint(0, num_classes, batch_size)
        
        top5_acc = top_k_accuracy(predictions, targets, k=5)
        
        self.assertGreaterEqual(top5_acc, 0)
        self.assertLessEqual(top5_acc, 1)


class TestVisualization(unittest.TestCase):
    """Test visualization utilities."""
    
    def test_plot_attention_heads(self):
        """Test attention visualization."""
        attention = np.random.randn(8, 10, 10)  # 8 heads, 10x10 attention
        tokens = [f"token_{i}" for i in range(10)]
        
        vis_data = plot_attention_heads(attention, tokens=tokens)
        
        self.assertEqual(vis_data['num_heads'], 8)
        self.assertEqual(vis_data['seq_len'], 10)
        self.assertIn('head_entropies', vis_data)
    
    def test_plot_embeddings(self):
        """Test embedding visualization."""
        embeddings = np.random.randn(100, 768)
        labels = np.random.randint(0, 10, 100)
        
        vis_data = plot_embeddings(embeddings, labels=labels, method='pca')
        
        self.assertEqual(vis_data['reduced'].shape[0], 100)
        self.assertEqual(vis_data['reduced'].shape[1], 2)
        self.assertIn('silhouette_score', vis_data)
    
    def test_plot_loss_curve(self):
        """Test loss curve visualization."""
        losses = [1.0, 0.9, 0.8, 0.75, 0.72, 0.70]
        
        vis_data = plot_loss_curve(losses, smoothing_window=2)
        
        self.assertEqual(len(vis_data['losses']), 6)
        self.assertLess(vis_data['final_loss'], vis_data['losses'][0])
    
    def test_plot_gradient_flow(self):
        """Test gradient flow visualization."""
        gradients = {
            'layer1': np.random.randn(100, 200),
            'layer2': np.random.randn(200, 50),
        }
        
        vis_data = plot_gradient_flow(gradients)
        
        self.assertIn('layer1', vis_data['layer_names'])
        self.assertIn('gradient_stats', vis_data)
    
    def test_plot_token_distribution(self):
        """Test token distribution visualization."""
        token_ids = np.random.randint(0, 1000, (100, 50))
        
        vis_data = plot_token_distribution(token_ids, top_k=10)
        
        self.assertEqual(len(vis_data['top_tokens']), 10)
        self.assertGreater(vis_data['entropy'], 0)
    
    def test_compare_attention_patterns(self):
        """Test attention pattern comparison."""
        attention1 = np.random.randn(10, 10)
        attention2 = np.random.randn(10, 10)
        
        comparison = compare_attention_patterns(attention1, attention2)
        
        self.assertIn('l1_difference', comparison)
        self.assertIn('cosine_similarity', comparison)


if __name__ == '__main__':
    unittest.main()
