import QtQuick.Window 2.2 //2.1
import QtQuick.Controls 1.4 //1.4
import QtQuick.Controls.Styles 1.4
import QtQuick.Extras 1.4
import QtQuick.Controls.Styles.Desktop 1.0
import QtQuick 2.12//2.12
import QtQuick.Window 2.12
import QtQuick.Controls 2.0
import QtQuick.Layouts 1.1
import QtLocation 5.11
import QtPositioning 5.0
import QtQuick.Window 2.3
import QtGraphicalEffects 1.0
import QtQuick.Controls.Imagine 2.3
import QtQuick.Controls.Material 2.0
import QtQuick 2.7

import QtQuick 2.12
import QtQuick.Window 2.13
import QtQuick.Controls 2.3
import QtQuick.Controls.Styles 1.4
import QtQuick.Extras 1.4
import QtQuick.Extras.Private 1.0

//import QtQuick.Window 2.2
import QtQuick 2.9


Window {
	
	id : root
	x : 30
	y : 30
	width: 1150
	maximumWidth : 1200
	minimumWidth : 1200
    height: 700
	maximumHeight : 700
	minimumHeight : 700
	title:"2D DPS CONTROLLER SIMULATOR"
	color : "#389BF2"
    visible: true

	property real sp_lat_val: -6.215861 // Variabel latitude
    property real sp_lon_val: 107.803706 // Variabel longitude

	Text {
		x : 1000
		y : 680
		width: 83
		height: 21
		color: "white"
		text: "(c) SYERGIE 2025"
		font.pixelSize: 14
		horizontalAlignment: Text.AlignHCenter
		verticalAlignment: Text.AlignVCenter
		font.family: "Verdana"
		font.bold: true
	}



	Rectangle{
		x : 950
		color : "transparent"
		border.width : 3
		border.color : "white"
		height : 650
		width : 250

	Text {
                id : starting_point
                x: 10
                y: 350
                width: 83
                height: 21
                color: "white"
                text: "starting point :"
                font.pixelSize: 14
                horizontalAlignment: Text.AlignLeft
                verticalAlignment: Text.AlignTop
                font.family: "Verdana"
                font.bold: true
				visible : true
            }



	Button {
           
            x: 10
            y: 400
            text : "set"
			width: 170
            //height: 31
            checkable: false
            checked: false
           
		   onClicked:{
						sp_lat_val = sp_lat.text
						sp_lon_val = sp_lon.text
						
						backend.setpoint(sp_lat.text, sp_lon.text, sp_yaw.text)
						//starting_point.text = sp_lat.text+","+sp_lon.text+","+sp_yaw.text
					
					}
			
        }


	Text {
		anchors.horizontalCenter: parent.horizontalCenter
		y : 20
		width: 83
		height: 21
		color: "white"
		text: "2D DPS CONTROLLER\nSIMULATOR"
		font.pixelSize: 14
		horizontalAlignment: Text.AlignHCenter
		verticalAlignment: Text.AlignVCenter
		font.family: "Verdana"
		font.bold: true
	}
			
	Text {
                
                x: 10
                y: 50
                width: 83
                height: 21
                color: "white"
                text: "Setpoint"
                font.pixelSize: 14
                horizontalAlignment: Text.AlignLeft
                verticalAlignment: Text.AlignTop
                font.family: "Verdana"
                font.bold: true
            }
			
	TextField{
					id : sp_lat
					x : 60
					y : 80
					text : "-6.215861"
					width : 120
					
					Text{
						//anchors.horizontalCenter: parent.horizontalCenter
						x : -50
						y:10
						text:"lat : "
						color : "white"
						font.family: "Cantora One"  // Set the font family
						font.pixelSize: 15    // Set the font size
						font.bold: true 
					}
				}
				
	TextField{
					id : sp_lon
					x : 60
					y : 120
					text : "107.803706"
					width : 120
					
					Text{
						//anchors.horizontalCenter: parent.horizontalCenter
						x : -50
						y:10
						text:"lon : "
						color : "white"
						font.family: "Cantora One"  // Set the font family
						font.pixelSize: 15    // Set the font size
						font.bold: true 
					}
				}
				
	TextField{
					id : sp_yaw
					x : 60
					y : 160
					text : "0"
					width : 120
					
					Text{
						//anchors.horizontalCenter: parent.horizontalCenter
						x : -50
						y:10
						text:"yaw : "
						color : "white"
						font.family: "Cantora One"  // Set the font family
						font.pixelSize: 15    // Set the font size
						font.bold: true 
					}
				}
			
			
			
			
	Text {
                
                x: 10
                y: 220
                width: 83
                height: 21
                color: "white"
                text: "DISTURBANCE :"
                font.pixelSize: 14
                horizontalAlignment: Text.AlignLeft
                verticalAlignment: Text.AlignTop
                font.family: "Verdana"
                font.bold: true
            }
			
			
	TextField{
					id : wind_speed
					x : 90
					y : 250
					text : "1"
					width : 120
					
					Text{
						//anchors.horizontalCenter: parent.horizontalCenter
						x : -80
						y:10
						text:"wind speed : "
						color : "white"
						font.family: "Cantora One"  // Set the font family
						font.pixelSize: 15    // Set the font size
						font.bold: true 
					}
				}
				
	TextField{
					id : wind_dir
					x : 90
					y : 300
					text :"90"
					width : 120
					
					Text{
						//anchors.horizontalCenter: parent.horizontalCenter
						x : -80
						y:10
						text:"wind dir : "
						color : "white"
						font.family: "Cantora One"  // Set the font family
						font.pixelSize: 15    // Set the font size
						font.bold: true 
					}
				}
			
	
	
	
	


	
	
	Button {
            id: controller_setup
            x: 10
            y: 90
            text : "controller setup"
			//width: 34
            //height: 31
            checkable: false
            checked: false
			visible : false
		   onClicked:{
						
					}
			
        }
 

}

	Rectangle{
		x : 525
		color : "transparent"
		border.width : 3
		border.color : "white"
		height : 530
		width : 420
		
		ScrollView {
			anchors.fill: parent
			clip: true
			ScrollBar.vertical.policy: ScrollBar.AlwaysOn
			contentHeight: 1000
			contentWidth: parent.width
		
		Text {
			anchors.horizontalCenter: parent.horizontalCenter
			y : 10
			width: 83
			height: 21
			color: "white"
			text: "STATE SPACE REPRESENTATION"
			font.pixelSize: 14
			horizontalAlignment: Text.AlignHCenter
			verticalAlignment: Text.AlignVCenter
			font.family: "Verdana"
			font.bold: true
		}
		
		
		Image{
			anchors.horizontalCenter: parent.horizontalCenter
			y : 40
			width : 300
			height : 150
			source : "state space.png"
		
		}
		
		
		Text {
			id : a
			x : 10
			y : 200
			width: 83
			height: 21
			color: "white"
			text: ""
			
			font.pixelSize: 9
			//horizontalAlignment: Text.AlignHCenter
			//verticalAlignment: Text.AlignVCenter
			font.family: "Verdana"
			
		}
		
		Text {
			id : b
			x : 10
			y : 300
			width: 83
			height: 21
			color: "white"
			text: ""
			font.pixelSize: 9
			//horizontalAlignment: Text.AlignHCenter
			//verticalAlignment: Text.AlignVCenter
			font.family: "Verdana"
			
		}
		
		
		Text {
			id : c
			x : 10
			y : 380
			width: 83
			height: 21
			color: "white"
			text: ""
			font.pixelSize: 9
			//horizontalAlignment: Text.AlignHCenter
			//verticalAlignment: Text.AlignVCenter
			font.family: "Verdana"
			
		}
		
		
		Text {
			id : x
			x : 150
			y : 380
			width: 83
			height: 21
			color: "white"
			text: ""
			font.pixelSize: 9
			//horizontalAlignment: Text.AlignHCenter
			//verticalAlignment: Text.AlignVCenter
			font.family: "Verdana"
			
		}
		
		
		Text {
			id : u
			x : 300
			y : 380
			width: 83
			height: 21
			color: "white"
			text: ""
			font.pixelSize: 9
			//horizontalAlignment: Text.AlignHCenter
			//verticalAlignment: Text.AlignVCenter
			font.family: "Verdana"
			
		}
		
		
		Text {
			id : y
			x : 10
			y : 430
			width: 83
			height: 21
			color: "white"
			text: ""
			font.pixelSize: 9
			//horizontalAlignment: Text.AlignHCenter
			//verticalAlignment: Text.AlignVCenter
			font.family: "Verdana"
			
		}
	
	}
		
	}

	Rectangle {
            id: map1
            x: 0
            y: 0
            width: 520
            height: 530
            color: "#958c8c"
            //radius: 6
            //border.color: "#6c6c6c"
            //border.width: 7
			
				Image {
					x : 10
					y : 540
					width : 300
					height : 150
					source : "thruster allocation.png"
				}
			
				Rectangle{
					x : 700
					y : 540
					width : 200
					height : 75
					color : "transparent"
					border.width : 3
					border.color : "white"
					
					Text {
							//anchors.horizontalCenter: parent.horizontalCenter
							x:80
							y: 10
							color: "white"
							text: "Propeller 1"
							font.pixelSize: 14
							horizontalAlignment: Text.AlignLeft
							verticalAlignment: Text.AlignTop
							font.family: "Verdana"
							font.bold: true
						}
					
					Image{
						id : azimuth1
						//anchors.horizontalCenter: parent.horizontalCenter
						x: 0
						y : 0
						width : 100
						height : 80
						source: "needlewhite.png"
						transformOrigin: Item.Center
						visible :true
						
						rotation: 180
						scale: 1
					}
					
					Text {
							//anchors.horizontalCenter: parent.horizontalCenter
							x:80
							id : properties1
							y: 50
							color: "white"
							text: "properties"
							font.pixelSize: 14
							horizontalAlignment: Text.AlignLeft
							verticalAlignment: Text.AlignTop
							font.family: "Verdana"
							
						}
					
					
				
				}

				Rectangle{
					x : 700
					y : 610
					width : 200
					height : 75
					color : "transparent"
					border.width : 3
					border.color : "white"
					
					Text {
							//anchors.horizontalCenter: parent.horizontalCenter
							
							x: 80
							y: 10
							color: "white"
							text: "Propeller 2"
							font.pixelSize: 14
							horizontalAlignment: Text.AlignLeft
							verticalAlignment: Text.AlignTop
							font.family: "Verdana"
							font.bold: true
						}
						
						
					Image{
						id : azimuth2
						//anchors.horizontalCenter: parent.horizontalCenter
						x : 0
						y : 0
						width : 100
						height : 80
						source: "needlewhite.png"
						transformOrigin: Item.Center
						visible :true
						
						rotation: 180
						scale: 1
					}
					
					Text {
							//anchors.horizontalCenter: parent.horizontalCenter
							id : properties2
							x : 80
							y: 50
							color: "white"
							text: "properties"
							font.pixelSize: 14
							horizontalAlignment: Text.AlignLeft
							verticalAlignment: Text.AlignTop
							font.family: "Verdana"
							
						}
					
					
				
				}



				Rectangle{
					x : 480
					y : 610
					width : 200
					height : 75
					color : "transparent"
					border.width : 3
					border.color : "white"
					
					Text {
							//anchors.horizontalCenter: parent.horizontalCenter
							x: 10
							y: 10
							color: "white"
							text: "Propeller 3"
							font.pixelSize: 14
							horizontalAlignment: Text.AlignLeft
							verticalAlignment: Text.AlignTop
							font.family: "Verdana"
							font.bold: true
						}
						
					Image{
						id : azimuth3
						//anchors.horizontalCenter: parent.horizontalCenter
						x : 120
						y : 0
						width : 100
						height : 80
						source: "needlewhite.png"
						transformOrigin: Item.Center
						visible :true
						
						rotation: 180
						scale: 1
					}
					
					Text {
							//anchors.horizontalCenter: parent.horizontalCenter
							x : 10
							id : properties3
							y: 50
							color: "white"
							text: "properties"
							font.pixelSize: 14
							horizontalAlignment: Text.AlignLeft
							verticalAlignment: Text.AlignTop
							font.family: "Verdana"
							
						}
					
					
				
				}



				Rectangle{
					x : 480
					y : 540
					width : 200
					height : 75
					color : "transparent"
					border.width : 3
					border.color : "white"
					
					Text {
							//anchors.horizontalCenter: parent.horizontalCenter
							
							x: 10
							y: 10
							color: "white"
							text: "Propeller 4"
							font.pixelSize: 14
							horizontalAlignment: Text.AlignLeft
							verticalAlignment: Text.AlignTop
							font.family: "Verdana"
							font.bold: true
						}
						
						
					Image{
						id : azimuth4
						//anchors.horizontalCenter: parent.horizontalCenter
						x : 120
						//y : 20
						width : 100
						height : 80
						source: "needlewhite.png"
						transformOrigin: Item.Center
						visible :true
						
						rotation: 180
						scale: 1
					}
					
					Text {
							//anchors.horizontalCenter: parent.horizontalCenter
							id : properties4
							x :10
							y: 50
							color: "white"
							text: "properties"
							font.pixelSize: 14
							horizontalAlignment: Text.AlignLeft
							verticalAlignment: Text.AlignTop
							font.family: "Verdana"
							
						}
					
					
				
				}

				Button{
					id : run
					x : 780
					y : 540
					width : 150
					height : 150
					checkable : true
					text : "run simulation"
					visible : false
					
				}
			
			
			
			
            gradient: Gradient {
                GradientStop {
                    position: 0
                    color: "#958c8c"
                }

                GradientStop {
                    position: 1
                    color: "#808080"
                }



            }
	
	
	
	
	
	Rectangle {
                id: mapGroup
                x: 0
                y: 0
                width: parent.width 
                height: parent.height
				
                property int count : 0
                property real lati : -6.000507
                property real longi : 106.687493	
				
				Map{
                    id: map
                    x: 0
                    y: 0
                    width: parent.width
                    height: parent.height
                    color: "#f9f9f9"
                    anchors.rightMargin: 8
                    anchors.centerIn: parent;
                    anchors.fill: parent
                    anchors.verticalCenterOffset: 0
                    anchors.horizontalCenterOffset: 0
                    anchors.bottomMargin: 0
                    anchors.top: parent.top
                    anchors.topMargin: 0
                    anchors.left: parent.left
                    anchors.leftMargin: 0
					zoomLevel : 100
                    maximumZoomLevel: 100.4
                    copyrightsVisible: true
                    antialiasing: true
                    maximumTilt: 89.3
                    plugin: mapPlugin
                    //activeMapType: supportedMapTypes[1]

                    center: QtPositioning.coordinate(latitude_position_value.text, longitude_position_value.text)

                    gesture.enabled: true
                    gesture.acceptedGestures: MapGestureArea.PinchGesture | MapGestureArea.PanGesture
				
				
				
				
						 
				Plugin {
				   id: mapPlugin
				   name: "osm"

				   //provide the address of the tile server to the plugin
				   PluginParameter {
					  name: "osm.mapping.custom.host"
					  value: "http://localhost/osm/"
				   }

				   /*disable retrieval of the providers information from the remote repository. 
				   If this parameter is not set to true (as shown here), then while offline, 
				   network errors will be generated at run time*/
				   PluginParameter {
					  name: "osm.mapping.providersrepository.disabled"
					  value: true
				   }
				}
				
				Line{
                    id: line
                }
                Line1{
                    id: line1
                }
				
				
				ListModel{
					id: md
				}
				
				ListModel{
					id: md1
				}
				
				


				
				Text {
                id: latitude_position_value
                x: 10
                y: 10
                width: 83
                height: 21
                color: "navy"
                text: qsTr("-2.75819")
                font.pixelSize: 14
                horizontalAlignment: Text.AlignLeft
                verticalAlignment: Text.AlignTop
                font.family: "Verdana"
                font.bold: true
            }

            Text {
                id: longitude_position_value
                x: 10
                y: 50
                width: 83
                height: 21
                color: "navy"
                text: qsTr("105.787")
                font.pixelSize: 14
                horizontalAlignment: Text.AlignLeft
                verticalAlignment: Text.AlignTop
                font.family: "Verdana"
                font.bold: true
            }
				
			
			
			
			
			
			
			
				MouseArea {
                        hoverEnabled: true
                        property var coordinate: map.toCoordinate(Qt.point(mouseX, mouseY))
                        x: 0
                        y: 0
                        width: parent.width
                        height: parent.height
                        

                        Label
                        {
                            x: parent.mouseX - width
                            y: parent.mouseY - height - 5
                            text: (parent.coordinate.latitude).toFixed(6) + "," +(parent.coordinate.longitude).toFixed(6)
                            color:"navy"

                        }
						
						
						Text{
						id : lat_mouse
						x: parent.mouseX - width
                        y: parent.mouseY - height - 5
						text: (parent.coordinate.latitude).toFixed(6)
						color : "red"
						visible : false
						
						}
					
						Text{
						id : long_mouse
						x: parent.mouseX - width
                        y: parent.mouseY - height - 5
						text: (parent.coordinate.longitude).toFixed(6)
						color : "red"
						visible : false
						
						}
						
						property var panjanglintasan: line.pathLength()
						property var path: line.path
						
                        onPressAndHold: {
                            //var crd = map.toCoordinate(Qt.point(mouseX, mouseY))
							
							
							console.log("clicked")
                            if (md.count < 1){
                                
                            }
                            else if (md.count > 0){
                                
                            }

                            markerModel.append({ "latitude":lat_mouse.text, "longitude": long_mouse.text})
                            var text = md.count + 1;
                            md.append({"coords": coordinate, "title": text})
                            line.addCoordinate(coordinate)

                            
                        }

                        onDoubleClicked: {
                            //var coor = map.toCoordinate(Qt.point(mouseX, mouseY))
                            //var text1 = md1.count + 1;
                            //md1.append({"coords": coordinate, "title": text1})
                            //line1.addCoordinate(coordinate)
                        }
						
						
						
                    }
					
				MapQuickItem {
					id: destination
					property alias text: txt.text
					sourceItem: Rectangle {
						width: 30
						height: 30
						color: "transparent"
						Image {
							anchors.fill: parent
							source: "cross_orange.png" // Ignore warnings from this
							sourceSize: Qt.size(parent.width, parent.height)
						}
						Text {
							id: txt
							anchors.fill: parent
						}
					}
					visible : true
					opacity: 1.0
					anchorPoint: Qt.point(sourceItem.width/2, sourceItem.height/2)
					coordinate: QtPositioning.coordinate(sp_lat_val, sp_lon_val)
				
				}

					
				
				MapQuickItem{
                    id : marker
                    sourceItem : Image{
                        id: imagenavigasi
                        width: 33
                        height: 37
                        //transformOrigin: Item.Center
                        source:"navigasi.png"
						//source:"segitiga.png"
                        //rotation: 0
                        fillMode: Image.PreserveAspectFit
                        transform: [
                            Rotation {
                                id: markerdirect
                                origin.x: 15
                                origin.y: 14
                                angle: 0
                            }]
                    }
					
					
					
                    coordinate: QtPositioning.coordinate(latitude_position_value.text, longitude_position_value.text)
                    //coordinate: QtPositioning.coordinate(2.73706666666667, 125.36065)
                    anchorPoint.x : 15
                    anchorPoint.y : 14
                    //anchorPoint.x : parent
                    //anchorPoint.y : parent

                }
				
				
				
				
				MapItemView {
                    id: mivMarker
                    model: ListModel {
                        id: markerModel
                    }
                    delegate: Component {
                        MapQuickItem {
                            coordinate: QtPositioning.coordinate(latitude, longitude)
                            property real slideIn: 0
                        }
                    } 
                }
				
				}
				
				
				
               

		   }


	
	}
	 
		
Timer{
		id:guitimer
		interval: 200
		repeat: true
		running: true
		onTriggered: {
			latitude_position_value.text = backend.latitude()
			longitude_position_value.text = backend.longitude()
			marker.rotation = backend.yaw()
			a.text = "A = " + backend.A_ss()
			b.text = "B = " + backend.B_ss()
			c.text = "C = " + backend.C_ss()
			x.text = "X = " + backend.x_ss()
			u.text = "U = " + backend.u_ss()
			y.text = "Y = " + backend.y_ss() + "\nY_ref = " + backend.yref_ss()
			
			properties1.text = backend.steering1() + "° / " + backend.gas_throttle1() + " N"
			properties2.text = backend.steering2() + "° / " + backend.gas_throttle2() + " N"
			properties3.text = backend.steering3() + "° / " + backend.gas_throttle3() + " N"
			properties4.text = backend.steering4() + "° / " + backend.gas_throttle4() + " N"
			
			azimuth1.rotation = backend.steering1()
			azimuth2.rotation = backend.steering2()
			azimuth3.rotation = backend.steering3()
			azimuth4.rotation = backend.steering4()
			
			starting_point.text = backend.start_lat() + " " + backend.start_lon() + "\n" + backend.delta_lat() + " " + backend.delta_lon() 
			if (run.checked == true){
				//animate.running = true
				backend.animate("1")
			
			}
			
			if (run.checked == false){
				//animate.running = false
				backend.animate("0")
			}
		}

}


Timer{
		id:animate
		interval: 1000
		repeat: true
		running: false
		onTriggered: {
			backend.animate("tick")
			
		}

}		
			
	
	
	
}













