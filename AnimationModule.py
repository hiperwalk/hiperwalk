"""
Module information
"""

from Constants import DEBUG
import matplotlib.pyplot as plt
from PIL import Image

if DEBUG:
    from time import time
    start = time()

class Animation:
    """
    Class responsible for creating, managing and saving animations.
    
    Attributes
    ----------
    frames : list
        List of frames.
    plt_anim : matplotlib.pyplot.Animation
        Animation object of matplotlib.
    save_path : str
        Filepath for saving an animation.
    """

    def __init__(self):
        self.frames = []
        self.plt_anim = None
        self.save_path = None

    def __del__(self):
        def _delete(obj):
            if obj is not None:
                del obj

        _delete(self.plt_anim)
        _delete(self.frames)
        _delete(self.save_path)

    def AddFrame(self, fig):
        """
        expects matplotlib fig
        storing images on RAM and clearing matplotlib image

        Parameters
        ----------
        fig
            matplotlib figure

        Examples
        --------
        Some stuff here
        """
        fig.canvas.draw()

        img = Image.frombytes('RGB', fig.canvas.get_width_height(),
            fig.canvas.tostring_rgb())
        self.frames.append(img)

        plt.close()

    def CreateAnimation(self, interval, repeat_delay):
        """
        TODO: repeat_delay not being used
        """
        from PlotModule import ConfigureFigure
        from matplotlib.animation import FuncAnimation

        fig, ax = ConfigureFigure(None)

        #it would be more efficient to use FuncAnimation as a previous version;
        #i.e. to return only the artist that need to be redrawn.
        #But workarounds for updating colorbar ticks and y-axes were not found
        #TODO: documentation: recommend saving gif and not showing for any animations
        def updateFigure(img):
            ax.imshow(img)

            if DEBUG:
                global start
                end = time()
                print('updateFigure: ' + str(end - start) + 's')
                print('\t' + str(img))
                start = end

        #TODO: repeat_delay not implemented in animation.save
        self.plt_anim = FuncAnimation(fig, updateFigure, frames=self.frames,
                interval=interval, repeat=self.__IsInNotebook())
        #repeat=True causes updateFigure to be called even when it is not needed.
        #However, it is necessary for creating a looping video in jupyter notebook

        #removing axes and pads introduced by imshow,
        #i.e. shows only the original axes and pads (from plots_as_imgs)
        plt.axis('off')
        plt.tight_layout(pad=0)
        if DEBUG:
            #used to check if no extra padding is being added
            fig.patch.set_facecolor('red')
            print("Fig dpi " + str(fig.dpi))

    #TODO: saving video is supported but not recommended because
    #   fps may not be respected.
    #ffmpeg is necessary for saving video and better quality gifs.
    #Pillow is sufficient for saving gifs, although colorbar may be discretized.
    def SaveAnimation(self, filename_prefix):
        valid_extensions = ['.gif', '.mp4'] #TODO: add other valid matplotlib formats

        extension = filename_prefix[:-4]
        if extension not in valid_extensions:
            filename_prefix += '.gif'
        self.save_path = filename_prefix

        self.plt_anim.save(self.save_path)
        plt.close()

        if DEBUG:
            print('finished saving')

        #by ignoring this condition while using the terminal,
        #a non-aborting exception is thrown:
        #AttributeError: 'NoneType' object has no attribute 'add_callback'.
        #The attributes deleted are needed for saving html5 video in jupyter notebooks
        if not self.__IsInNotebook():
            del self.plt_anim
            self.plt_anim = None
            del self.frames
            self.frames = None

    def __IsSaved(self):
        return self.save_path != None

    #returns None if animation is already saved.
    #Otherwise returns the temporary file
    def __SaveAnimationInTempFile(self):
        if not self.__IsSaved():
            import tempfile
            temp = tempfile.NamedTemporaryFile(suffix='.gif')
            self.SaveAnimation(temp.name)
            return temp

        return None

    def __IsInNotebook(self):
        try:
            shell = get_ipython().__class__.__name__
            if shell == 'ZMQInteractiveShell':
                return True
            return False
        except:
            return False

    def ShowAnimation(self):
        """
        Shows animation as long as it has been created.

        See Also
        --------
        CreateAnimation
        """
        if self.__IsInNotebook():
            self.__ShowAnimationNotebook()
        else:
            self.__ShowAnimationTerminal()

    def __ShowAnimationTerminal(self):
        temp = self.__SaveAnimationInTempFile()

        from gi import require_version
        require_version('Gtk', '3.0')

        from gi.repository import Gtk as gtk
        from gi.repository.Gdk import KEY_q
        from gi.repository.Gdk import KEY_Q
        from gi.repository.Gdk import RGBA
        
        #creating window
        window = gtk.ApplicationWindow(title="Animation")
        state = window.get_state()
        window.override_background_color(state, RGBA(1, 1, 1, 1))

        def ConfigureGifWindow(self):
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

        def ConfigureVideoWindow(self):
            print('Warning: show video not supported.')

        if self.save_path[-4:] == '.gif':
            ConfigureGifWindow(self)
        else:
            ConfigureVideoWindow(self)


        #showing window and starting gtk main loop
        window.show_all()
        gtk.main()
        
        if temp is not None:
            self.save_path = None
            temp.close()

    def __ShowAnimationNotebook(self):
        from IPython import display

        #embedding animation in jupyter notebook
        video = self.plt_anim.to_html5_video()
        html = display.HTML(video)
        display.display(html)

        plt.close()

