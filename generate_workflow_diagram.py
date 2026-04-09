# generate_workflow_diagram.py
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def create_workflow_diagram():
    fig, ax = plt.subplots(1, 1, figsize=(12, 12))  # figura más alta
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 12)  # más alto
    ax.axis('off')

    # Definir los pasos con mayor espaciado vertical
    steps = [
        (5, 11.0, "Typology Prediction\n(Random Forest)", "#d9eaf7"),
        (5, 9.0, "Unit Replacement Cost\n(USD/m²)\n← Local cost database", "#fdebd0"),
        (5, 7.0, "Building Area (m²) × level\n← from footprint", "#d5f5e3"),
        (5, 5.0, "Replacement Value =\nUnit Cost × Area", "#fdebd0"),
        (5, 3.0, "Age Category\n(decade: 1980, 1990, 2000)\n← predicted by model", "#d9eaf7"),
        (5, 1.0, "Depreciation Factor\n← local age‑depreciation curve", "#fdebd0"),
        (5, -1.0, "Current Built‑Up Valuation =\nReplacement Value × Depreciation Factor", "#d5f5e3"),
    ]

    # Dibujar recuadros y flechas
    for i, (x, y, text, color) in enumerate(steps):
        rect = patches.FancyBboxPatch((x-3, y-0.7), 6, 1.4,
                                      boxstyle="round,pad=0.2",
                                      facecolor=color, edgecolor="black",
                                      linewidth=1.5)
        ax.add_patch(rect)
        ax.text(x, y, text, ha='center', va='center', fontsize=10, fontweight='bold')

        # Flecha hacia abajo (excepto última)
        if i < len(steps)-1:
            arrow = patches.FancyArrowPatch((x, y-0.7), (x, steps[i+1][1]+0.7),
                                            arrowstyle='->', mutation_scale=20,
                                            linewidth=2, color='gray')
            ax.add_patch(arrow)

    ax.text(5, 12.7, "From Typology to Current Built‑Up Valuation",
            ha='center', va='top', fontsize=14, fontweight='bold')

    # Nota aclaratoria
    ax.text(5, -1.8, "All economic values must be calibrated locally (Central‑Pampean Argentina values are examples only)",
            ha='center', va='center', fontsize=8, style='italic', color='gray')

    plt.tight_layout()
    plt.savefig("workflow_diagram.png", dpi=300, bbox_inches='tight')
    plt.show()
    print("Diagram saved as 'workflow_diagram.png'")

if __name__ == "__main__":
    create_workflow_diagram()