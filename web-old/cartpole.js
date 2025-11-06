// CartPole WebAssembly Visualization
import init, { WasmCartPole } from './thrust_rl.js';

// Canvas and rendering
const canvas = document.getElementById('cartpole-canvas');
const ctx = canvas.getContext('2d');

// Stats elements
const episodeEl = document.getElementById('episode');
const stepsEl = document.getElementById('steps');
const bestScoreEl = document.getElementById('best-score');

// Control buttons
const startBtn = document.getElementById('start-btn');
const pauseBtn = document.getElementById('pause-btn');
const resetBtn = document.getElementById('reset-btn');
const speedSlider = document.getElementById('speed-slider');
const speedValueEl = document.getElementById('speed-value');

// Environment and state
let env = null;
let isRunning = false;
let animationId = null;
let stepsPerFrame = 1;

// Physics constants (matching Gym CartPole-v1)
const CART_WIDTH = 50;
const CART_HEIGHT = 30;
const POLE_WIDTH = 10;
const POLE_LENGTH = 100;
const TRACK_Y = 400;

// Rendering constants
const X_SCALE = 100; // pixels per meter
const X_CENTER = canvas.width / 2;

// Initialize WASM and environment
async function initialize() {
    await init();
    env = new WasmCartPole();
    env.reset();
    updateStats();
    render();
}

// Update stats display
function updateStats() {
    episodeEl.textContent = env.get_episode();
    stepsEl.textContent = env.get_steps();
    bestScoreEl.textContent = env.get_best_score();
}

// Random policy (for demonstration)
function randomAction() {
    return Math.random() < 0.5 ? 0 : 1;
}

// Main game loop
function gameLoop() {
    if (!isRunning) return;

    for (let i = 0; i < stepsPerFrame; i++) {
        // Get action (random policy for now)
        const action = randomAction();

        // Step environment
        const result = env.step(action);

        // result format: [obs0, obs1, obs2, obs3, reward, terminated, truncated]
        const terminated = result[5] === 1.0;
        const truncated = result[6] === 1.0;

        if (terminated || truncated) {
            env.reset();
            updateStats();
        }
    }

    updateStats();
    render();
    animationId = requestAnimationFrame(gameLoop);
}

// Render the CartPole environment
function render() {
    // Clear canvas
    ctx.fillStyle = '#f8f9fa';
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    // Get state [x, x_dot, theta, theta_dot]
    const state = env.get_state();
    const [x, x_dot, theta, theta_dot] = state;

    // Draw track
    ctx.strokeStyle = '#495057';
    ctx.lineWidth = 3;
    ctx.beginPath();
    ctx.moveTo(50, TRACK_Y);
    ctx.lineTo(canvas.width - 50, TRACK_Y);
    ctx.stroke();

    // Calculate cart position
    const cartX = X_CENTER + x * X_SCALE;
    const cartY = TRACK_Y;

    // Draw cart
    ctx.fillStyle = '#4263eb';
    ctx.fillRect(
        cartX - CART_WIDTH / 2,
        cartY - CART_HEIGHT,
        CART_WIDTH,
        CART_HEIGHT
    );

    // Draw pole
    ctx.save();
    ctx.translate(cartX, cartY - CART_HEIGHT / 2);
    ctx.rotate(theta); // theta is already in radians

    ctx.fillStyle = '#fa5252';
    ctx.fillRect(
        -POLE_WIDTH / 2,
        -POLE_LENGTH,
        POLE_WIDTH,
        POLE_LENGTH
    );

    // Draw pole joint
    ctx.fillStyle = '#343a40';
    ctx.beginPath();
    ctx.arc(0, 0, 8, 0, Math.PI * 2);
    ctx.fill();

    ctx.restore();

    // Draw state info
    ctx.fillStyle = '#212529';
    ctx.font = '14px monospace';
    ctx.fillText(`x: ${x.toFixed(3)}`, 10, 20);
    ctx.fillText(`θ: ${theta.toFixed(3)}`, 10, 40);
    ctx.fillText(`ẋ: ${x_dot.toFixed(3)}`, 10, 60);
    ctx.fillText(`θ̇: ${theta_dot.toFixed(3)}`, 10, 80);
}

// Event handlers
startBtn.addEventListener('click', () => {
    if (!isRunning) {
        isRunning = true;
        startBtn.disabled = true;
        pauseBtn.disabled = false;
        gameLoop();
    }
});

pauseBtn.addEventListener('click', () => {
    isRunning = false;
    startBtn.disabled = false;
    pauseBtn.disabled = true;
    if (animationId) {
        cancelAnimationFrame(animationId);
        animationId = null;
    }
});

resetBtn.addEventListener('click', () => {
    env.reset();
    updateStats();
    render();
});

speedSlider.addEventListener('input', (e) => {
    stepsPerFrame = parseInt(e.target.value);
    speedValueEl.textContent = `${stepsPerFrame}x`;
});

// Initialize on page load
initialize().catch(console.error);
