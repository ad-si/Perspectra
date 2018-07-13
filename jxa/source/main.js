/* globals ObjC, $, Library */
/* eslint-disable new-cap */

const config = require('./config')

function bitOr (elements) {
  return elements.reduce((current, next) => {
    // eslint-disable-next-line no-bitwise
    return current | next
  })
}

ObjC.import('Cocoa')

const styleMask = bitOr([
  $.NSTitledWindowMask,
  $.NSClosableWindowMask,
  $.NSMiniaturizableWindowMask,
  $.NSResizableWindowMask,
])

if (!$.MyWindow) {
  ObjC.registerSubclass({
    name: 'MyWindow',
    superclass: 'NSWindow',
    methods: {
      mouseDown: {
        types: ['void', ['id']],
        implementation: () => {
          $.NSLog('Left mouse click')
        },
      },
      rightMouseDown: {
        types: ['void', ['id']],
        implementation: () => {
          $.NSLog('Right mouse click')
        },
      },
    },
  })
}

const appWindow = $.MyWindow.alloc.initWithContentRectStyleMaskBackingDefer(
  $.NSMakeRect(0, 0, config.appWindow.width, config.appWindow.height),
  styleMask,
  $.NSBackingStoreBuffered,
  false
)

function chooseImage () {
  const panel = $.NSOpenPanel.openPanel
  panel.title = 'Select an Image'
  panel.allowedFileTypes = $(config.allowedTypes)

  if (panel.runModal === $.NSOKButton) {
    // Panel.URLs is an NSArray not a JS array
    const imagePath = panel.URLs.objectAtIndex(0).path
    textField.stringValue = imagePath

    const image = $.NSImage.alloc.initByReferencingFile(imagePath)
    const imageRect = $.NSMakeRect(
      0,
      config.appWindow.height,
      image.size.width,
      image.size.height
    )
    const imageView = $.NSImageView.alloc.initWithFrame(imageRect)
    const width = image.size.width > config.appWindow.minWidth
      ? image.size.width
      : config.appWindow.minWidth
    const height = (
      image.size.height > config.appWindow.minHeight
        ? image.size.height
        : config.appWindow.minHeight
    ) + config.appWindow.controlBarHeight

    appWindow.setFrameDisplay(
      $.NSMakeRect(0, 0, width, height),
      true
    )

    imageView.setImage(image)
    appWindow.contentView.addSubview(imageView)
  }
}


if (!$.AppDelegate) {
  ObjC.registerSubclass({
    name: 'AppDelegate',
    methods: {
      btnClickHandler: {
        types: ['void', ['id']],
        implementation: chooseImage,
      },
    },
  })
}
const appDelegate = $.AppDelegate.alloc.init


const textFieldLabelRect = $.NSMakeRect(
  25,
  config.appWindow.height - 40,
  config.textField.label.width,
  config.textField.label.height
)
const textFieldLabel = $.NSTextField.alloc.initWithFrame(textFieldLabelRect)
Object.assign(textFieldLabel, config.textField.properties)
appWindow.contentView.addSubview(textFieldLabel)


const textFieldRect = $.NSMakeRect(
  25,
  config.appWindow.height - 60,
  config.textField.width,
  config.textField.height
)
const textField = $.NSTextField.alloc.initWithFrame(textFieldRect)
textField.editable = false
appWindow.contentView.addSubview(textField)

const buttonRect = $.NSMakeRect(
  230,
  config.appWindow.height - 62,
  config.button.width,
  config.button.height
)
const button = $.NSButton.alloc.initWithFrame(buttonRect)
button.title = 'Choose an Image …'
button.bezelStyle = $.NSRoundedBezelStyle
button.buttonType = $.NSMomentaryLightButton
button.target = appDelegate
button.action = 'btnClickHandler'
appWindow.contentView.addSubview(button)



appWindow.center
appWindow.title = 'Perspectra'
appWindow.makeKeyAndOrderFront(appWindow)
