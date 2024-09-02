import torch
import torch.nn as nn
import timm


class Geoformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model('convnextv2_base', pretrained=True, num_classes=0)

        dim = self._get_dim()
        print(f"dim: {dim}")

        self.shared_layer = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Dropout(0.1)
        )

        self.lat_lon = nn.Linear(dim, 2)

        self.quadtree = nn.Linear(dim, 11399)

        
    def _get_dim(self):
        samp = torch.rand(1, 3, 224, 224)
        with torch.no_grad():
            out = self.backbone(samp)
        return out.shape[1]
        
    def forward(self, x):
        x = self.backbone(x)
        x = self.shared_layer(x)
        lat_lon = self.lat_lon(x)
        
        outputs = {
            "latitude": lat_lon[:, 0],
            "longitude": lat_lon[:, 1],
            "quadtree_10_1000": self.quadtree(x),
        }

        return outputs
    

import PySimpleGUI as sg
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import cartopy.crs as ccrs
import cartopy.feature as cfeature

def draw_figure(canvas, figure):
    figure_canvas_agg = FigureCanvasTkAgg(figure, canvas)
    figure_canvas_agg.draw()
    figure_canvas_agg.get_tk_widget().pack(side='top', fill='both', expand=1)
    return figure_canvas_agg

def create_map(latitude, longitude, zoom_level=6):
    fig, ax = plt.subplots(figsize=(10, 8), subplot_kw={'projection': ccrs.PlateCarree()})
    
    ax.set_extent([longitude-zoom_level, longitude+zoom_level, 
                   latitude-zoom_level, latitude+zoom_level], crs=ccrs.PlateCarree())
    
    ax.add_feature(cfeature.LAND)
    ax.add_feature(cfeature.OCEAN)
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    
    ax.plot(longitude, latitude, 'ro', markersize=10, transform=ccrs.Geodetic())
    
    return fig

def main():
    sg.theme('DefaultNoMoreNagging')

    layout = [
        [sg.Canvas(key='-CANVAS-')],
        [sg.Button('Zoom In'), sg.Button('Zoom Out'), sg.Button('Exit')]
    ]

    window = sg.Window('Zoomable Map', layout, finalize=True, element_justification='center', font='Helvetica 18')

    latitude, longitude = 40.7128, -74.0060  # New York City
    zoom_level = 6
    fig = create_map(latitude, longitude, zoom_level)
    fig_agg = draw_figure(window['-CANVAS-'].TKCanvas, fig)

    while True:
        event, values = window.read()
        if event in (sg.WIN_CLOSED, 'Exit'):
            break
        elif event == 'Zoom In':
            zoom_level = max(1, zoom_level - 1)
        elif event == 'Zoom Out':
            zoom_level += 1

        plt.close(fig)
        fig = create_map(latitude, longitude, zoom_level)
        fig_agg.get_tk_widget().forget()
        fig_agg = draw_figure(window['-CANVAS-'].TKCanvas, fig)

    window.close()

if __name__ == '__main__':
    main()