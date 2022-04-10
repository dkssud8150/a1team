<div align="center">

 <br>
   
# devcorse a1_team 이치현 / 윤재호

</div>

<br>

hough변환을 사용하여 라인을 검출하였다. 그 결과를 통해 pid제어를 추가했다. pid제어 값을 조정하고 있는 중이다.

---

## 현재 상황
1. pid제어를 하고 있다.
2. 최대한 많은 pid 파라미터를 직접 부여하여 작동시켜보았다.
3. 차선을 검출할 위치도 지정해주는 것이 좋기 때문에, 그에 대한 `offset` 파라미터를 조정하는 것도 중요하다.

---

## 목표
1. pid값 찾기
2. 주행에 필요한 pid값을 찾고 속도에 대해서도 pid를 적용
3. 차선을 인식할 코드에 대해 더 좋은 방법이 있는지 찾아보기
4. 차선을 인식하지 못했을 때(차선이 끊김, 급커브로 인해 차선이 한족이 없어짐 등)에 대한 처리도 추가 

<br>

---


<br>

<details open>
   <summary>🚀 Stack</summary>
<code><img alt = "3.1 Python" height="20" src="https://cdn.icon-icons.com/icons2/2699/PNG/512/pytorch_logo_icon_170820.png"> python </code>
<code><img alt = "3.1 Python" height="20" src="https://cdn.icon-icons.com/icons2/171/PNG/512/xml_23331.png"> xml </code>
<code><img alt = "3.1 Python" height="20" src="https://cdn.icon-icons.com/icons2/665/PNG/512/robot_icon-icons.com_60269.png"> ROS </code> 
</details>
 
<!-- xycar_ws 파일 다운로드 링크
https://drive.google.com/drive/folders/14LWo5XMtEs9XdGgKs_X4Z2-8SkvMUuMs?usp=sharing --> 
