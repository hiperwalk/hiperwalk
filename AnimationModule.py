from Constants import DEBUG
import matplotlib.pyplot as plt
from PIL import Image

if DEBUG:
    from time import time
    start = time()

class Animation:

    def __init__(self):
        self.frames = []
        self.plt_anim = None
        self.save_path = None

    #expects matplotlib fig
    #storing images on RAM and clearing matplotlib image
    def AddFrame(self, fig):
        fig.canvas.draw()

        img = Image.frombytes('RGB', fig.canvas.get_width_height(),
            fig.canvas.tostring_rgb())
        self.frames.append(img)

        plt.close()



    #TODO: repeat_delay not being used
    def CreateAnimation(self, interval, repeat_delay):
        from PlotModule import ConfigureFigure
        from matplotlib.animation import FuncAnimation

        fig, ax = ConfigureFigure(None)

        #it would be more efficient to use FuncAnimation as a previous version;
        #i.e. to return only the artist that need to be redrawn.
        #But workarounds for updating colorbar ticks and y-axes were not found
        #TODO: documentation: recommend saving gif and not showing for any animations
        def updateFigure(img):
            ax.imshow(img, animated=True)

            if DEBUG:
                global start
                end = time()
                print('updateFigure: ' + str(end - start) + 's')
                start = end

        #TODO: repeat_delay not implemented in animation.save
        self.plt_anim = FuncAnimation(fig, updateFigure, frames=self.frames,
                interval=interval, repeat_delay=repeat_delay)

        #removing axes and pads introduced by imshow,
        #i.e. shows only the original axes and pads (from plots_as_imgs)
        plt.axis('off')
        plt.tight_layout(pad=0)
        if DEBUG:
            #used to check if no extra padding is being added
            fig.patch.set_facecolor('red')
            print("Fig dpi " + str(fig.dpi))

    def SaveAnimation(self, filename_prefix):
        self.save_path = filename_prefix + '.gif'
        self.plt_anim.save(self.save_path)
        plt.clf()

    def ShowAnimation(self):
        from gi.repository import Gtk as gtk
        from gi.repository.Gdk import KEY_q
        from gi.repository.Gdk import KEY_Q
        from gi.repository.Gdk import RGBA
        
        #creating window
        window = gtk.Window(title="Animation")
        state = window.get_state()
        window.override_background_color(state, RGBA(1, 1, 1, 1))

        #assign closing events
        def on_key_press(self, event):
            if event.keyval == KEY_q or event.keyval == KEY_Q :
                window.close()

        window.connect("key-press-event", on_key_press)
        window.connect("destroy", gtk.main_quit)

        #exhibits animation in gtk window
        img = gtk.Image.new()
        img.set_from_file(self.save_path)
        window.add(img)

        #showing window and starting gtk main loop
        window.show_all()
        gtk.main()
