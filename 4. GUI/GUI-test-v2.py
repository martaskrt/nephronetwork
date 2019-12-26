import PySimpleGUI as SG

sag_path = SG.PopupGetFile('Sagittal', 'Choose a file')
print(sag_path)

trans_path = SG.PopupGetFile('Transverse', 'Choose a file')
print(trans_path)

## parse file based on file ending


SG.popup('You entered the following filenames\n',
         'Sagittal: ',sag_path, '\n',
         'Transverse: ',trans_path)

window  = SG.Window('Radio Button',[SG.Radio('My first Radio!     ', "RADIO1", default=True, size=(10,1))])


