# Web Visualizer Implementation Plan

## Tech Stack
- **Framework**: React 19 + TypeScript + Vite
- **Styling**: Tailwind CSS v4
- **Linting**: Biome
- **Routing**: React Router
- **WASM**: Rust environments compiled to WebAssembly
- **2D Rendering**: Pixi.js v8 (for Snake)
- **3D Rendering**: Three.js + React Three Fiber (for CartPole)

## Architecture

```
web/
├── src/
│   ├── components/
│   │   ├── ui/              # shadcn/ui components
│   │   ├── CartPole/
│   │   │   ├── CartPole3D.tsx       # Three.js 3D renderer
│   │   │   ├── CartPoleControls.tsx # UI controls
│   │   │   └── useCartPole.ts       # WASM integration hook
│   │   └── Snake/
│   │       ├── SnakePixi.tsx        # Pixi.js 2D renderer
│   │       ├── SnakeControls.tsx    # UI controls
│   │       └── useSnake.ts          # WASM integration hook
│   ├── lib/
│   │   ├── wasm.ts          # WASM loader utilities
│   │   └── utils.ts         # Helper functions
│   ├── pages/
│   │   ├── Home.tsx         # Landing page
│   │   ├── CartPolePage.tsx
│   │   └── SnakePage.tsx
│   └── App.tsx
└── public/
    ├── thrust_rl_bg.wasm
    ├── thrust_rl.js
    └── assets/              # Sprites, textures
```

## Implementation Phases

### Phase 1: Foundation (30 min)
- [x] Vite + React + TypeScript setup
- [x] Tailwind CSS v4 configuration
- [x] Biome setup
- [x] WASM files in public/
- [ ] Add Pixi.js and Three.js dependencies
- [ ] Set up React Router
- [ ] Create basic page structure
- [ ] Add shadcn/ui components (Button, Card, Slider)

### Phase 2: Snake with Pixi.js (1-2 hours)
- [ ] Install @pixi/react for React integration
- [ ] Create SnakePixi component
  - Grid rendering with sprites
  - Snake body segments with textures
  - Food particles with glow effect
  - Smooth movement interpolation
- [ ] Create useSnake hook
  - WASM environment initialization
  - Game loop with requestAnimationFrame
  - State management (running, paused, reset)
- [ ] Snake controls UI
  - Start/Pause/Reset buttons
  - Speed slider (1x-10x)
  - Stats display (episode, steps, alive agents)
- [ ] Visual polish
  - Death animation with particle explosion
  - Food collection effect
  - Agent color coding
  - Grid fade effect

### Phase 3: CartPole with Three.js (1-2 hours)
- [ ] Install @react-three/fiber and @react-three/drei
- [ ] Create CartPole3D component
  - 3D cart mesh (box geometry)
  - 3D pole mesh (cylinder geometry)
  - Track plane
  - Lighting (ambient + directional)
  - Shadows
  - Camera controls (OrbitControls)
- [ ] Create useCartPole hook
  - WASM environment initialization
  - Game loop with useFrame
  - Physics state update
- [ ] CartPole controls UI
  - Start/Pause/Reset buttons
  - Speed slider
  - Stats display (episode, steps, best score)
  - Camera reset button
- [ ] Visual polish
  - Pole falling animation
  - Success/failure visual feedback
  - Environment lighting changes
  - Optional: Cart trail effect

### Phase 4: Landing Page (30 min)
- [ ] Hero section with gradient background
- [ ] Demo cards for CartPole and Snake
- [ ] Feature highlights
- [ ] GitHub link
- [ ] Responsive design

### Phase 5: Polish & Optimization (1 hour)
- [ ] Loading states with suspense
- [ ] Error boundaries
- [ ] Performance monitoring
- [ ] Mobile responsiveness
- [ ] Accessibility (keyboard controls)
- [ ] SEO meta tags

### Phase 6: Build & Deploy (30 min)
- [ ] Configure Vite for GitHub Pages
- [ ] Build optimization
- [ ] Test production build
- [ ] Deploy to GitHub Pages
- [ ] Set up custom domain (optional)

## Visual Design Goals

### Snake (Pixi.js)
- **Style**: Modern, neon aesthetic
- **Colors**: Vibrant agent colors (red, blue, green, orange)
- **Effects**:
  - Glow on snakes and food
  - Particle trails
  - Death explosions
  - Food collection sparkles
- **Performance**: Solid 60 FPS with 4 agents × 50 segments

### CartPole (Three.js)
- **Style**: Clean, scientific visualization
- **Materials**:
  - Metallic cart with PBR materials
  - Glossy pole
  - Matte track
- **Lighting**:
  - Ambient for base lighting
  - Directional with shadows for depth
  - Optional: Spotlight following cart
- **Camera**:
  - Default: Side view showing full motion
  - User-controlled orbit
  - Smooth transitions

## Performance Targets
- **Initial load**: < 2s on 3G
- **WASM initialization**: < 500ms
- **Frame rate**: Stable 60 FPS
- **Memory**: < 100MB total
- **Bundle size**: < 500KB (gzipped, excluding WASM)

## Accessibility
- Keyboard controls for all interactions
- ARIA labels on all interactive elements
- Screen reader announcements for game state
- Reduced motion option
- High contrast mode support

## Browser Support
- Chrome/Edge 90+
- Firefox 88+
- Safari 14+
- Mobile browsers (iOS Safari, Chrome Android)
