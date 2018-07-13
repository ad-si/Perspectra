const config = {
  allowedFileTypes: ['jpg', 'png', 'gif'],
}

Object.assign(config, {
  appWindow: {
    width: 600,
    height: 85,
    minWidth: 400,
    minHeight: 340,
    controlBarHeight: 80,
  },
  textField: {
    width: 205,
    height: 24,
    label: {
      width: 200,
      height: 24,
    },
    properties: {
      stringValue: `Image: (${config.allowedFileTypes.join(', ')})`,
      drawsBackground: false,
      editable: false,
      bezeled: false,
      selectable: true,
    },
  },
  button: {
    width: 150,
    height: 25,
    title: 'Choose an WHR …',
  },
})

module.exports = config
