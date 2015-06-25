import sys, time
class progressBar:
    """ Creates a text-based progress bar. Call the object with the `print'
        command to see the progress bar, which looks something like this:

        [=======>        22%                  ]

        You may specify the progress bar's width, min and max values on init.
    """

    def __init__(self, minValue = 0, maxValue = 100, totalWidth=65):
        self.progBar = "[]"   # This holds the progress bar string
        self.min = minValue
        self.max = maxValue
        self.span = maxValue - minValue
        self.width = totalWidth
        self.amount = 0       # When amount == max, we are 100% done
        self.updateAmount(0)  # Build progress bar string
        self.start = time.time()

    def updateAmount(self, newAmount = 0):
        """ Update the progress bar with the new amount (with min and max
            values set at initialization; if it is over or under, it takes the
            min or max value as a default. """
        if newAmount < self.min: newAmount = self.min
        if newAmount > self.max: newAmount = self.max
        self.amount = newAmount
        self.now = time.time()

        # Figure out the new percent done, round to an integer
        diffFromMin = float(self.amount - self.min)
        percentDone = (diffFromMin / float(self.span)) * 100.0
        percentDone = int(round(percentDone))

        # Figure out how many hash bars the percentage should be
        allFull = self.width - 2
        numHashes = (percentDone / 100.0) * allFull
        numHashes = int(round(numHashes))

        # Build a progress bar with an arrow of equal signs; special cases for
        # empty and full
        if numHashes == 0:
            self.progBar = "[>%s]" % (' '*(allFull-1))
        elif numHashes == allFull:
            self.progBar = "[%s]" % ('='*allFull)
        else:
            self.progBar = "[%s>%s]" % ('='*(numHashes-1),
                                        ' '*(allFull-numHashes))

        # figure out where to put the percentage, roughly centered
        percentPlace = (len(self.progBar) / 2) - len(str(percentDone))
        percentString = str(percentDone) + "%"

        # slice the percentage into the bar
        self.progBar = ''.join([self.progBar[0:percentPlace], percentString,
                                self.progBar[percentPlace+len(percentString):] ])
        #add estimated time left
        #try:
        if not float(self.amount-self.min) == 0.:
            esl = float(self.max - self.amount)/(self.amount-self.min) * (self.now - self.start) 
            hours = int(esl / 3600)
            minutes = int((esl % 3600)/60)
            seconds = int(esl - hours*3600 - minutes*60)
        else:
            hours = 0
            minutes = 0
            seconds = 0
        self.progBar += 'ETL:%ih%im%is' % (hours, minutes, seconds)

    def __str__(self):
        return str(self.progBar)

    def __call__(self, value):
        """ Updates the amount, and writes to stdout. Prints a carriage return
            first, so it will overwrite the current line in stdout."""
        print '\r',
        self.updateAmount(value)
        sys.stdout.write(str(self))
        sys.stdout.flush()

try:
    import Tkinter

    class PB:
        
        def settitle(self, title):
            self.__root.title(title)
            
            
        # Create Progress Bar
        def __init__(self, width, height):
            #self.__root = Tkinter.Toplevel()
            self.__root = Tkinter.Tk() #updated by Petr
            self.__root.resizable(False, False)
            self.__root.title('Wait please...')
            self.__canvas = Tkinter.Canvas(self.__root, width=width, height=height)
            self.__canvas.grid()
            self.__width = width
            self.__height = height

        # Open Progress Bar
        def open(self):
            self.__root.deiconify()
            self.__root.focus_set()
            #self.__root.update()

        # Close Progress Bar
        def close(self):
            self.__root.withdraw()

        # Update Progress Bar
        def update(self, ratio):
            self.__canvas.delete(Tkinter.ALL)
            self.__canvas.create_rectangle(0, 0, self.__width * ratio, \
                                           self.__height, fill='blue')
            self.__root.update()
            self.__root.focus_set()
except:pass
