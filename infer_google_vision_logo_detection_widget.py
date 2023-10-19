from ikomia import core, dataprocess
from ikomia.utils import pyqtutils, qtconversion
from infer_google_vision_logo_detection.infer_google_vision_logo_detection_process import InferGoogleVisionLogoDetectionParam

# PyQt GUI framework
from PyQt5.QtWidgets import *


# --------------------
# - Class which implements widget associated with the algorithm
# - Inherits PyCore.CWorkflowTaskWidget from Ikomia API
# --------------------
class InferGoogleVisionLogoDetectionWidget(core.CWorkflowTaskWidget):

    def __init__(self, param, parent):
        core.CWorkflowTaskWidget.__init__(self, parent)

        if param is None:
            self.parameters = InferGoogleVisionLogoDetectionParam()
        else:
            self.parameters = param

        # Create layout : QGridLayout by default
        self.grid_layout = QGridLayout()
        # PyQt -> Qt wrapping
        layout_ptr = qtconversion.PyQtToQt(self.grid_layout)


        # Confidence threshold
        self.spin_conf_thres = pyqtutils.append_double_spin(
            self.grid_layout,
            "Confidence threshold",
            self.parameters.conf_thres,
            min=0.,
            max=1.,
            step=0.01,
            decimals=2
        )

        # Credentials
        self.browse_credentials = pyqtutils.append_browse_file(
                                            grid_layout=self.grid_layout,
                                            label="Google app credentials (.json)",
                                            path=self.parameters.google_application_credentials,
                                            mode=QFileDialog.ExistingFile
        )
 
        # Set widget layout
        self.set_layout(layout_ptr)

    def on_apply(self):
        # Apply button clicked slot
        self.parameters.conf_thres = self.spin_conf_thres.value()
        self.parameters.google_application_credentials = self.browse_credentials.path

        # Send signal to launch the algorithm main function
        self.emit_apply(self.parameters) 


# --------------------
# - Factory class to build algorithm widget object
# - Inherits PyDataProcess.CWidgetFactory from Ikomia API
# --------------------
class InferGoogleVisionLogoDetectionWidgetFactory(dataprocess.CWidgetFactory):

    def __init__(self):
        dataprocess.CWidgetFactory.__init__(self)
        # Set the algorithm name attribute -> it must be the same as the one declared in the algorithm factory class
        self.name = "infer_google_vision_logo_detection"

    def create(self, param):
        # Create widget object
        return InferGoogleVisionLogoDetectionWidget(param, None)
