// Multi-Agent Snake WebAssembly Visualization
import init, { WasmSnake } from './thrust_rl.js';

// Canvas and rendering
const canvas = document.getElementById('snake-canvas');
const ctx = canvas.getContext('2d');

// Stats elements
const episodeEl = document.getElementById('episode');
const stepsEl = document.getElementById('steps');
const aliveEl = document.getElementById('alive');

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
let stepCount = 0;

// Grid settings
const GRID_WIDTH = 20;
const GRID_HEIGHT = 20;
const CELL_SIZE = canvas.width / GRID_WIDTH;
const NUM_AGENTS = 4;

// Agent colors
const AGENT_COLORS = [
    '#e74c3c', // Red
    '#3498db', // Blue
    '#2ecc71', // Green
    '#f39c12', // Orange
];

const FOOD_COLOR = '#9b59b6'; // Purple
const GRID_COLOR = '#ecf0f1';

// Initialize WASM and environment
async function initialize() {
    await init();
    env = new WasmSnake(GRID_WIDTH, GRID_HEIGHT, NUM_AGENTS);
    env.reset();
    updateStats();
    render();
}

// Update stats display
function updateStats() {
    episodeEl.textContent = env.get_episode();
    stepsEl.textContent = stepCount;

    // Count alive agents
    const activeAgents = env.active_agents();
    const aliveCount = activeAgents.filter(a => a === 1).length;
    aliveEl.textContent = `${aliveCount}/${NUM_AGENTS}`;
}

// Random policy (for demonstration)
function randomAction() {
    return Math.floor(Math.random() * 4); // 0=Up, 1=Down, 2=Left, 3=Right
}

// Main game loop
function gameLoop() {
    if (!isRunning) return;

    for (let i = 0; i < stepsPerFrame; i++) {
        // Get actions for all agents (random policy for now)
        const actions = [];
        for (let j = 0; j < NUM_AGENTS; j++) {
            actions.push(randomAction());
        }

        // Step environment
        env.step(actions);
        stepCount++;

        // Check if all agents are dead
        const activeAgents = env.active_agents();
        const aliveCount = activeAgents.filter(a => a === 1).length;

        if (aliveCount === 0) {
            env.reset();
            stepCount = 0;
            updateStats();
        }
    }

    updateStats();
    render();
    animationId = requestAnimationFrame(gameLoop);
}

// Render the Snake environment
function render() {
    // Clear canvas
    ctx.fillStyle = '#ffffff';
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    // Draw grid
    ctx.strokeStyle = GRID_COLOR;
    ctx.lineWidth = 1;
    for (let x = 0; x <= GRID_WIDTH; x++) {
        ctx.beginPath();
        ctx.moveTo(x * CELL_SIZE, 0);
        ctx.lineTo(x * CELL_SIZE, canvas.height);
        ctx.stroke();
    }
    for (let y = 0; y <= GRID_HEIGHT; y++) {
        ctx.beginPath();
        ctx.moveTo(0, y * CELL_SIZE);
        ctx.lineTo(canvas.width, y * CELL_SIZE);
        ctx.stroke();
    }

    // Get food positions
    const foodPositions = env.get_food_positions();
    for (let i = 0; i < foodPositions.length; i += 2) {
        const x = foodPositions[i];
        const y = foodPositions[i + 1];
        drawCell(x, y, FOOD_COLOR);

        // Draw food as circle
        ctx.fillStyle = FOOD_COLOR;
        ctx.beginPath();
        ctx.arc(
            (x + 0.5) * CELL_SIZE,
            (y + 0.5) * CELL_SIZE,
            CELL_SIZE * 0.3,
            0,
            Math.PI * 2
        );
        ctx.fill();
    }

    // Get snake positions and draw
    // Format: [agent0_len, agent0_x0, agent0_y0, agent0_x1, agent0_y1, ..., agent1_len, ...]
    const snakePositions = env.get_snake_positions();
    const activeAgents = env.active_agents();

    let offset = 0;
    for (let agent = 0; agent < NUM_AGENTS; agent++) {
        if (offset >= snakePositions.length) break;

        const length = snakePositions[offset];
        offset++;

        const isAlive = activeAgents[agent] === 1;
        const color = isAlive ? AGENT_COLORS[agent] : '#95a5a6';

        // Draw snake body
        for (let i = 0; i < length; i++) {
            if (offset + 1 >= snakePositions.length) break;

            const x = snakePositions[offset];
            const y = snakePositions[offset + 1];
            offset += 2;

            // Draw body segment
            drawCell(x, y, color, i === 0 ? 1.0 : 0.7);

            // Draw head differently (first segment)
            if (i === 0) {
                // Draw eyes on head
                ctx.fillStyle = '#ffffff';
                const eyeSize = CELL_SIZE * 0.15;
                const eyeOffset = CELL_SIZE * 0.25;
                ctx.beginPath();
                ctx.arc(
                    (x + 0.5 - eyeOffset) * CELL_SIZE,
                    (y + 0.4) * CELL_SIZE,
                    eyeSize,
                    0,
                    Math.PI * 2
                );
                ctx.arc(
                    (x + 0.5 + eyeOffset) * CELL_SIZE,
                    (y + 0.4) * CELL_SIZE,
                    eyeSize,
                    0,
                    Math.PI * 2
                );
                ctx.fill();
            }
        }
    }
}

// Draw a single grid cell
function drawCell(gridX, gridY, color, alpha = 1.0) {
    const x = gridX * CELL_SIZE;
    const y = gridY * CELL_SIZE;

    ctx.fillStyle = color;
    ctx.globalAlpha = alpha;
    ctx.fillRect(
        x + 1,
        y + 1,
        CELL_SIZE - 2,
        CELL_SIZE - 2
    );
    ctx.globalAlpha = 1.0;
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
    stepCount = 0;
    updateStats();
    render();
});

speedSlider.addEventListener('input', (e) => {
    stepsPerFrame = parseInt(e.target.value);
    speedValueEl.textContent = `${stepsPerFrame}x`;
});

// Initialize on page load
initialize().catch(console.error);
