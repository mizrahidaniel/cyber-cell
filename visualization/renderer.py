"""Taichi GUI rendering with cell display, chemical overlays, and stats."""

import taichi as ti

from config import (
    GRID_WIDTH, GRID_HEIGHT, GUI_SCALE, MAX_CELLS, DAY_LENGTH,
)
from cell.cell_state import (
    cell_alive, cell_x, cell_y, cell_energy, cell_genome_id,
    grid_cell_id, cell_count,
)
from cell.genome import genome_count
from world.grid import light_field

# Display buffer: RGB float image
display = ti.Vector.field(3, dtype=ti.f32, shape=(GRID_WIDTH, GRID_HEIGHT))

# Overlay mode: 0=cells+light, 1=S, 2=R, 3=G
_overlay_mode = 0


@ti.func
def genome_to_color(gid: ti.i32) -> ti.math.vec3:
    """Hash genome ID to an HSV hue, convert to RGB."""
    # Simple hash for hue (0-1 range)
    h = ti.cast((gid * 7919 + 104729) % 360, ti.f32) / 360.0
    # HSV to RGB (S=0.8, V varies with energy)
    s = 0.8
    v = 1.0

    c = v * s
    h6 = h * 6.0
    x = c * (1.0 - ti.abs(h6 % 2.0 - 1.0))
    m = v - c

    r = m
    g = m
    b = m
    if h6 < 1.0:
        r += c; g += x
    elif h6 < 2.0:
        r += x; g += c
    elif h6 < 3.0:
        g += c; b += x
    elif h6 < 4.0:
        g += x; b += c
    elif h6 < 5.0:
        r += x; b += c
    else:
        r += c; b += x

    return ti.math.vec3(r, g, b)


@ti.kernel
def render_cells_and_light():
    """Render cells colored by genome, background shows light intensity."""
    for i, j in display:
        # Light background (subtle gray)
        light = light_field[i, j] * 0.15
        bg = ti.math.vec3(light, light, light * 0.8)

        cid = grid_cell_id[i, j]
        if cid >= 0 and cell_alive[cid] == 1:
            color = genome_to_color(cell_genome_id[cid])
            # Brightness scales with energy (dim when low, bright when high)
            brightness = ti.min(1.0, cell_energy[cid] / 100.0) * 0.7 + 0.3
            display[i, j] = color * brightness
        else:
            display[i, j] = bg


@ti.kernel
def render_chemical_overlay(chem: ti.template(), channel: ti.i32):
    """Render a chemical field as a colored heatmap."""
    for i, j in display:
        val = ti.min(1.0, chem[i, j] * 2.0)  # scale for visibility
        light = light_field[i, j] * 0.08

        r = light
        g = light
        b = light

        if channel == 0:  # S = green
            g += val
        elif channel == 1:  # R = red
            r += val
        elif channel == 2:  # G (signal) = blue
            b += val

        # Also show cells as white dots
        cid = grid_cell_id[i, j]
        if cid >= 0 and cell_alive[cid] == 1:
            r = 1.0
            g = 1.0
            b = 1.0

        display[i, j] = ti.math.vec3(r, g, b)


class Renderer:
    def __init__(self):
        self.gui = ti.GUI("CyberCell",
                          res=(GRID_WIDTH * GUI_SCALE, GRID_HEIGHT * GUI_SCALE),
                          fast_gui=False)
        self.overlay_mode = 0
        self.paused = False
        self.fast_forward = False

    def handle_input(self) -> bool:
        """Process keyboard input. Returns False if window should close."""
        for e in self.gui.get_events(ti.GUI.PRESS):
            if e.key == ti.GUI.ESCAPE:
                return False
            elif e.key == '1':
                self.overlay_mode = 0
            elif e.key == '2':
                self.overlay_mode = 1
            elif e.key == '3':
                self.overlay_mode = 2
            elif e.key == '4':
                self.overlay_mode = 3
            elif e.key == ' ':
                self.paused = not self.paused
            elif e.key == 'f':
                self.fast_forward = not self.fast_forward
        return True

    def render(self, tick: int, env_S, env_R, env_G):
        """Render one frame."""
        if self.overlay_mode == 0:
            render_cells_and_light()
        elif self.overlay_mode == 1:
            render_chemical_overlay(env_S, 0)
        elif self.overlay_mode == 2:
            render_chemical_overlay(env_R, 1)
        elif self.overlay_mode == 3:
            render_chemical_overlay(env_G, 2)

        self.gui.set_image(display)

        # Stats text
        pop = cell_count[None]
        gen = genome_count[None]
        day_phase = "DAY" if (tick % DAY_LENGTH) < DAY_LENGTH // 2 else "NIGHT"
        mode_names = ["Cells", "Structure(S)", "Replication(R)", "Signal(G)"]

        self.gui.text(f"Tick: {tick}  {day_phase}", pos=(0.01, 0.98), color=0xFFFFFF)
        self.gui.text(f"Pop: {pop}  Genomes: {gen}", pos=(0.01, 0.95), color=0xFFFFFF)
        self.gui.text(f"View: {mode_names[self.overlay_mode]} [1-4]", pos=(0.01, 0.92), color=0xAAAAAA)
        self.gui.text("[Space]=Pause [F]=Fast [Esc]=Quit", pos=(0.01, 0.89), color=0x888888)

        self.gui.show()

    def close(self):
        self.gui.close()
